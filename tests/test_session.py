"""Session-level tests for the solicited-response matcher and the staged wait."""

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from yalexs_ble import util
from yalexs_ble.const import Commands, SettingType
from yalexs_ble.lock import (
    _ack_matcher,
    _operation_response_matcher,
    _settings_response_matcher,
)
from yalexs_ble.session import (
    RESPONSE_TIMEOUT,
    AuthError,
    DisconnectedError,
    OperationIncompleteError,
    OperationProgress,
    ResponseError,
    Session,
)

# Verbatim field frames (2026-07-16 capture, YUR/DEL fw 2.1.0): the READSETTING
# acknowledgment and, ~40 ms later, the 0xBB frame carrying the stored value
# (Timed 1800: both uint16 timers set to 1800).
READ_ACK = bytes.fromhex("aa0400282800000000000000000000000200")
READ_ANSWER = bytes.fromhex("bb0400fb2800000008070807000000000000")
# The same value frame with its checksum byte cleared, so _validate_response
# rejects it the way a corrupted frame off the air is rejected.
CORRUPT_ANSWER = bytes.fromhex("bb0400002800000008070807000000000000")


def _make_matcher_session(received: list[bytes]) -> Session:
    """Create a Session with a mock client and pass-through crypto."""
    client = MagicMock(is_connected=True)
    session = Session(client, "testlock", asyncio.Lock(), set(), received.append)
    session.decrypt = bytes  # type: ignore[method-assign, assignment]
    session.cipher_encrypt = MagicMock(update=bytes)
    return session


@pytest.mark.asyncio
async def test_settings_wait_skips_ack_and_completes_on_value_frame() -> None:
    """With a matcher, the acknowledgment does not answer the command.

    The 0xAA acknowledgment arrives first and must leave the wait armed (its
    zero value field would decode as auto-lock off); the 0xBB settings frame
    resolves it. Both frames still reach the state callback.
    """
    received: list[bytes] = []
    session = _make_matcher_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )

    async def deliver(*_args: object, **_kwargs: object) -> None:
        # The wait is armed before the GATT write, so both response frames
        # can be delivered from here, in their on-air order.
        session._notify(0, bytearray(READ_ACK))
        # The acknowledgment must not have resolved the wait.
        assert session._notify_future is not None
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status", matcher)

    assert result == READ_ANSWER
    assert received == [READ_ACK, READ_ANSWER]


@pytest.mark.asyncio
async def test_execute_without_matcher_takes_first_valid_frame() -> None:
    """Without a matcher the first valid frame answers, as before."""
    received: list[bytes] = []
    session = _make_matcher_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ACK))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status")

    assert result == READ_ACK


@pytest.mark.asyncio
async def test_corrupt_frame_disarms_the_wait_and_the_command_is_retried() -> None:
    """A frame that fails the checksum ends the wait, and the write is repeated.

    The corrupt frame resolves the future with a ResponseError, so the matcher
    must be cleared with it -- otherwise the retry re-arms the wait with the
    previous command's matcher still in place. The staged operation wait skips
    corrupt frames instead; this is the plain path, where re-writing a settings
    command costs nothing.
    """
    received: list[bytes] = []
    session = _make_matcher_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )
    frames = [CORRUPT_ANSWER, READ_ANSWER]

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(frames.pop(0)))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session.execute(bytearray(18), "auto_lock_status", matcher)

    assert result == READ_ANSWER
    assert session.client.write_gatt_char.await_count == 2
    # Both frames reached the state callback; only the valid one answered.
    assert received == [CORRUPT_ANSWER, READ_ANSWER]


@pytest.mark.asyncio
async def test_corrupt_frame_on_every_attempt_raises_and_leaves_the_slot_empty() -> (
    None
):
    """Three corrupt frames exhaust the retries and the error reaches the caller."""
    received: list[bytes] = []
    session = _make_matcher_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(CORRUPT_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    with pytest.raises(ResponseError):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session.client.write_gatt_char.await_count == 3
    assert session._notify_future is None
    assert session._response_matcher is None


def _short_timeout(_seconds: float) -> object:
    """Replacement for util.asyncio_timeout that expires almost immediately."""
    return asyncio.timeout(0.01)


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_success() -> None:
    """On the happy path the notify slot is empty when _locked_write returns."""
    received: list[bytes] = []
    session = _make_matcher_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session._locked_write(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session._notify_future is None
    assert session._response_matcher is None


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_timeout() -> None:
    """When no response arrives, the timeout path disarms the notify slot."""
    received: list[bytes] = []
    session = _make_matcher_session(received)
    # The write succeeds but no notify is ever delivered.
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    assert session._response_matcher is None


@pytest.mark.asyncio
async def test_late_notify_after_timeout_is_ignored() -> None:
    """A frame arriving after the timeout cleared the slot is a no-op.

    The timeout leaves ``_notify_future`` at None, so a late frame reaches the
    state callback but resolves no wait and raises nothing.
    """
    received: list[bytes] = []
    session = _make_matcher_session(received)
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    session._notify(0, bytearray(READ_ANSWER))
    assert session._notify_future is None
    # The late frame still reached the state callback.
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_fresh_command_rearms_slot_after_timeout() -> None:
    """After a timeout, a fresh command re-arms the slot and completes normally."""
    received: list[bytes] = []
    session = _make_matcher_session(received)
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")
    assert session._notify_future is None

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session._locked_write(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session._notify_future is None


def _with_checksum(hex_str: str) -> bytearray:
    """Build an 18-byte frame with a valid simple checksum in byte[3].

    _validate_response requires the 18-byte simple checksum to sum to zero;
    byte[3] is the checksum field, so set it to whatever makes that hold.
    """
    frame = bytearray.fromhex(hex_str)
    frame[0x03] = 0
    frame[0x03] = util._simple_checksum(frame)
    return frame


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Yield to the event loop until predicate() holds (bounded).

    Lets the operation under test advance to its next awaited stage before the
    test feeds the next frame, so each _notify lands in the intended stage.
    """
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


def _make_session(
    state_callback: Callable[[bytes], None] | None = None,
) -> tuple[Session, MagicMock]:
    """Build a Session over a mock client for the notify/staged-wait path.

    Only cipher_encrypt is set (execute_operation asserts it); cipher_decrypt
    is left None so notify frames pass through Session.decrypt unmodified.
    """
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    session = Session(client, "mylock", asyncio.Lock(), set(), state_callback)
    session.cipher_encrypt = Cipher(
        algorithms.AES(bytes(16)),
        modes.CBC(bytes(16)),
    ).encryptor()
    return session, client


# securemode = LOCK opcode (0x0B) with operation byte 0x04.
_ACK_SECUREMODE = "aa0b00000400000000000000000000000200"
_OP_RESPONSE_OK = "bb0b00000000000000000000000000000200"
_SETTLED_STATUS = "bb0200000200000000000000000000000200"
_FOREIGN_ACK = "aa0b00000000000000000000000000000200"


@pytest.mark.asyncio
async def test_execute_operation_happy_path() -> None:
    """Write, then ack, then op-response: returns the op-response bytes.

    write_success_callback fires once, before the ack is delivered.
    """
    order: list[str] = []
    session, _ = _make_session()
    write_cb = MagicMock(side_effect=lambda: order.append("write"))
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        order.append("ack")
        session._notify(0, bytearray(ack))
        # Deliver the op-response in the SAME event-loop turn: both stage
        # futures resolve before the writer resumes, exercising the
        # supersede branch, which must still record the acknowledgement.
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
        write_success_callback=write_cb,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.command_written is True
    assert progress.acknowledged is True
    write_cb.assert_called_once()
    # The callback fired before the ack was delivered.
    assert order == ["write", "ack"]


@pytest.mark.asyncio
async def test_execute_operation_ignores_foreign_frames() -> None:
    """A settle and a foreign ack mid-wait do not complete the operation."""
    seen: list[bytes] = []
    session, _ = _make_session(seen.append)
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    settled = _with_checksum(_SETTLED_STATUS)
    foreign_ack = _with_checksum(_FOREIGN_ACK)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        # A settled GETSTATUS (0xBB 0x02) and a foreign ack (0xAA 0x0B with
        # operation byte 0x00, not the securemode 0x04) must not complete it.
        session._notify(0, bytearray(settled))
        await asyncio.sleep(0)
        session._notify(0, bytearray(foreign_ack))
        await asyncio.sleep(0)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    assert progress.acknowledged is True
    # The foreign frames were still handed to the state callback.
    assert bytes(settled) in seen
    assert bytes(foreign_ack) in seen


@pytest.mark.asyncio
async def test_execute_operation_op_response_before_ack() -> None:
    """The op-response alone completes the wait; the ack stage is superseded."""
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    # No acknowledgement was needed to complete.
    assert progress.acknowledged is False


@pytest.mark.asyncio
async def test_execute_operation_ack_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No acknowledgement within the ack budget raises a plain TimeoutError."""
    monkeypatch.setattr("yalexs_ble.session.RESPONSE_TIMEOUT", 0.05)
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)

    with pytest.raises(TimeoutError) as exc_info:
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )

    # A plain TimeoutError, not the non-retryable OperationIncompleteError:
    # nothing was acknowledged, so the command may still be retried.
    assert not isinstance(exc_info.value, OperationIncompleteError)
    assert progress.command_written is True
    assert progress.acknowledged is False


@pytest.mark.asyncio
async def test_execute_operation_op_response_timeout() -> None:
    """Acknowledged but no op-response raises the non-retryable error."""
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        session._notify(0, bytearray(ack))

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=0.1,
            progress=progress,
        )
    await feeder

    assert progress.acknowledged is True


@pytest.mark.asyncio
async def test_stage_deadlines_run_from_the_command_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stage timeouts subtract the time already spent in the attempt.

    The acknowledgement wait asks for RESPONSE_TIMEOUT minus the time the
    write consumed, and the op-response wait asks for response_timeout minus
    everything spent up to the acknowledgement, so the whole operation is
    bounded from the moment the command is issued, exactly as
    OperationIncompleteError's message states.
    """
    session, client = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)
    op_response = _with_checksum(_OP_RESPONSE_OK)

    clock = {"now": 1000.0}
    monkeypatch.setattr("yalexs_ble.session.time.monotonic", lambda: clock["now"])

    async def write_taking_half_a_second(*args: object, **kwargs: object) -> None:
        clock["now"] += 0.5

    client.write_gatt_char = AsyncMock(side_effect=write_taking_half_a_second)

    ack_wait_timeouts: list[float | None] = []
    real_wait = asyncio.wait

    async def spying_wait(fs: object, **kwargs: object) -> object:
        ack_wait_timeouts.append(kwargs.get("timeout"))  # type: ignore[arg-type]
        return await real_wait(fs, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("yalexs_ble.session.asyncio.wait", spying_wait)

    op_wait_timeouts: list[float] = []
    real_asyncio_timeout = util.asyncio_timeout

    def spying_asyncio_timeout(delay: float) -> object:
        op_wait_timeouts.append(delay)
        return real_asyncio_timeout(delay)

    monkeypatch.setattr(
        "yalexs_ble.session.util.asyncio_timeout", spying_asyncio_timeout
    )

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        clock["now"] += 2.5  # the acknowledgement lands 3.0 s into the attempt
        session._notify(0, bytearray(ack))
        # The op-response must land in the op-response stage, so wait for that
        # stage to arm its own bound rather than for progress.acknowledged,
        # which is set as the frame is received and so is already true here.
        await _spin_until(lambda: len(op_wait_timeouts) == 2)
        session._notify(0, bytearray(op_response))

    feeder = asyncio.create_task(feed())
    result = await session.execute_operation(
        command,
        "force_securemode",
        ack_matcher=_ack_matcher(0x0B, 0x04),
        response_matcher=_operation_response_matcher(0x0B),
        response_timeout=5.0,
        progress=progress,
    )
    await feeder

    assert result == bytes(op_response)
    # The write consumed 0.5 s of the attempt, so the acknowledgement stage
    # asked for the remainder of RESPONSE_TIMEOUT, not the full budget again.
    assert ack_wait_timeouts == [pytest.approx(RESPONSE_TIMEOUT - 0.5)]
    # The first bound is the write stage's own RESPONSE_TIMEOUT. The
    # acknowledgement then spent 3.0 s of the 5.0 s response budget, so the
    # op-response stage asked for the 2.0 s remainder.
    assert op_wait_timeouts == [
        pytest.approx(RESPONSE_TIMEOUT),
        pytest.approx(2.0),
    ]


@pytest.mark.asyncio
async def test_execute_with_response_matcher_skips_nonmatching_and_bad_frames() -> None:
    """The plain execute path waits past a non-matching and a garbled frame."""
    session, _ = _make_session()
    command = session.build_command(Commands.GETSTATUS)

    def matcher(data: bytes) -> bool:
        return len(data) > 1 and data[0] == 0xBB and data[1] == 0x02

    nonmatching = _with_checksum(_OP_RESPONSE_OK)  # valid 0xBB 0x0B, wrong op
    good = _with_checksum(_SETTLED_STATUS)  # valid 0xBB 0x02, matches
    bad = _with_checksum(_SETTLED_STATUS)
    bad[0x03] = (bad[0x03] + 1) & 0xFF  # corrupt the checksum

    async def feed() -> None:
        await _spin_until(lambda: session._notify_future is not None)
        session._notify(0, bytearray(nonmatching))
        await asyncio.sleep(0)
        # A garbled frame during a matcher wait must not error or re-send;
        # the wait simply continues.
        session._notify(0, bytearray(bad))
        await asyncio.sleep(0)
        session._notify(0, bytearray(good))

    feeder = asyncio.create_task(feed())
    result = await session.execute(command, "lock_status", matcher)
    await feeder

    assert result == bytes(good)


# =========================================================================== #
# Disconnect / auth contract of the operation wait (guards b81d75446c62)
#
# The pre/post-acknowledgement retry hinge: a failure BEFORE the ack keeps its
# retryable type (the command never moved the motor); once acknowledged the
# result is unknown, so a disconnect or timeout becomes the non-retryable
# OperationIncompleteError and the command is never silently re-sent. These
# tests pin that contract so a future change that makes an acknowledged
# operation retryable again fails the suite instead of shipping.
# =========================================================================== #
def _fire_disconnect(session: Session) -> None:
    """Resolve the operation's disconnected future, as a link drop would.

    execute_operation registers a future in _disconnected_futures and wraps the
    wait in async_interrupt.interrupt(...); resolving that future raises
    DisconnectedError inside the wait.
    """
    for fut in list(session._disconnected_futures):
        if not fut.done():
            fut.set_result(None)


@pytest.mark.asyncio
async def test_execute_operation_disconnect_after_ack_is_operation_incomplete() -> None:
    """Disconnect AFTER the ack -> non-retryable OperationIncompleteError."""
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        session._notify(0, bytearray(ack))
        await _spin_until(lambda: progress.acknowledged)
        _fire_disconnect(session)  # link drops while awaiting the op-response

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder


@pytest.mark.asyncio
async def test_ack_and_disconnect_in_one_turn_is_operation_incomplete() -> None:
    """Ack and disconnect in ONE event-loop turn -> non-retryable.

    The interrupt cancels the staged wait before it resumes, so nothing in the
    wait body observes the acknowledgement. It still has to be recorded, or the
    drop is classified as a pre-acknowledgement disconnect and the command is
    re-sent to a lock that already took it.
    """
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)
    ack = _with_checksum(_ACK_SECUREMODE)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        # Both resolutions land in the same turn, with no await between them.
        session._notify(0, bytearray(ack))
        _fire_disconnect(session)

    feeder = asyncio.create_task(feed())
    with pytest.raises(OperationIncompleteError):
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder

    assert progress.acknowledged is True


@pytest.mark.asyncio
async def test_execute_operation_disconnect_before_ack_stays_retryable() -> None:
    """Disconnect BEFORE the ack -> plain DisconnectedError (retryable)."""
    session, _ = _make_session()
    progress = OperationProgress()
    command = session.build_operation_command(Commands.LOCK, 0x04)

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        _fire_disconnect(session)  # link drops before any acknowledgement

    feeder = asyncio.create_task(feed())
    with pytest.raises(DisconnectedError) as exc_info:
        await session.execute_operation(
            command,
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )
    await feeder

    # Retryable: not the non-retryable type, and nothing was acknowledged.
    assert not isinstance(exc_info.value, OperationIncompleteError)
    assert progress.acknowledged is False


@pytest.mark.asyncio
async def test_execute_operation_key_error_on_write_raises_auth_error() -> None:
    """A key/slot error on the first request surfaces as AuthError."""
    session, client = _make_session()
    client.write_gatt_char = AsyncMock(side_effect=BleakError("Unlikely Error"))

    with pytest.raises(AuthError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=OperationProgress(),
        )


@pytest.mark.asyncio
async def test_execute_operation_not_connected_raises_disconnected() -> None:
    """The pre-write connected-guard raises, converted to DisconnectedError."""
    session, client = _make_session()
    client.is_connected = False

    with pytest.raises(DisconnectedError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=OperationProgress(),
        )


@pytest.mark.asyncio
async def test_bleak_disconnect_after_ack_is_operation_incomplete() -> None:
    """Defensive branch: a BleakError disconnect surfacing AFTER the ack is also
    treated as an unknown result (OperationIncompleteError), not a retry."""
    session, _ = _make_session()
    progress = OperationProgress()

    def _boom(
        command: bytearray,
        command_name: str,
        ack_matcher: Callable[[bytes], bool],
        response_matcher: Callable[[bytes], bool],
        response_timeout: float,
        progress: OperationProgress,
        write_success_callback: Callable[[], None] | None = None,
    ) -> bytes:
        progress.acknowledged = True
        raise BleakError("device disconnected")

    with (
        patch.object(session, "_locked_write_operation", AsyncMock(side_effect=_boom)),
        pytest.raises(OperationIncompleteError),
    ):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=progress,
        )


@pytest.mark.asyncio
async def test_execute_settings_not_connected_raises_disconnected() -> None:
    """The settings/poll path's pre-write connected-guard also raises."""
    session, client = _make_session()
    client.is_connected = False

    with pytest.raises(DisconnectedError):
        await session.execute(session.build_command(Commands.GETSTATUS), "status")


@pytest.mark.asyncio
async def test_notify_returns_when_only_ack_is_armed_and_frame_does_not_match() -> None:
    """A frame that matches neither the armed ack nor a (None) result wait is
    dropped without resolving anything -- the guard for a result wait that is
    not armed."""
    session, _ = _make_session()
    session._ack_future = asyncio.get_running_loop().create_future()
    session._ack_matcher = _ack_matcher(0x0B, 0x04)
    session._notify_future = None
    session._response_matcher = None

    # A settled GETSTATUS (0xBB) never matches the 0xAA ack matcher.
    session._notify(0, bytearray(_with_checksum(_SETTLED_STATUS)))

    assert not session._ack_future.done()


@pytest.mark.asyncio
async def test_execute_operation_generic_bleak_error_is_reraised() -> None:
    """A BleakError that is neither a key nor a disconnect error re-raises as-is.

    It stays a plain (retryable) BleakError -- not converted to AuthError or
    DisconnectedError -- so the write may simply be retried.
    """
    session, client = _make_session()
    client.write_gatt_char = AsyncMock(side_effect=BleakError("GATT write failed"))

    with pytest.raises(BleakError):
        await session.execute_operation(
            session.build_operation_command(Commands.LOCK, 0x04),
            "force_securemode",
            ack_matcher=_ack_matcher(0x0B, 0x04),
            response_matcher=_operation_response_matcher(0x0B),
            response_timeout=5.0,
            progress=OperationProgress(),
        )


# =========================================================================== #
# 1000 ms field replays -- the typed staged wait vs the wrong-capture frames
#
# Verbatim, ordered field frame sequences from fixtures_1000ms_wrong_captures.md
# where the OLD untyped wait captured a wrong frame, written ahead of the staged
# wait that answers them. Each is replayed through session._notify during an
# execute_operation staged wait. Frame classes:
#   * bb0a / bb0b = 0xBB op-response (opcode 0x0a unlock / 0x0b lock+securemode)
#   * bb02        = 0xBB 0x02 settled GETSTATUS status push
#   * aa0a / aa0b = 0xAA acknowledgement echoing the written opcode
# byte[3] checksums are valid as captured, so the frames are fed unchanged.
# =========================================================================== #
@pytest.mark.asyncio
async def test_replay_incident1_force_lock_ignores_prev_unlock_stale_bb0a() -> None:
    """Incident 1: a force_lock wait ignores the previous unlock's stale bb0a.

    The old untyped wait was satisfied by bb0a003b... -- the delayed op-response
    (opcode 0x0a = unlock) of the PRECEDING force_unlock, a leftover frame, not
    an answer to this force_lock (opcode 0x0b). The typed wait keeps it on the
    state-callback path and completes only on this force_lock's own bb0b.
    """
    seen: list[bytes] = []
    session, _ = _make_session(seen.append)
    progress = OperationProgress()
    command = session.build_command(Commands.LOCK)  # force_lock: opcode 0x0b

    # Verbatim field frames (byte[3] checksums valid as captured).
    stale_unlock_result = bytes.fromhex("bb0a003b0000000000000000000000000000")
    settled_unlocked = bytes.fromhex("bb02003e0200000003000000000000000000")
    true_lock_ack = bytes.fromhex("aa0b00490000000000000000000000000200")
    genuine_lock_result = bytes.fromhex("bb0b003a0000000000000000000000000000")

    op_task: asyncio.Task[bytes] | None = None

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        # The previous unlock's stale op-response and a settled status push:
        # neither is this force_lock's ack or op-response.
        session._notify(0, bytearray(stale_unlock_result))
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert op_task is not None and not op_task.done()  # wrong frame ignored
        # The genuine LOCK ack (opcode 0x0b, op byte 0x00) advances to stage 2.
        session._notify(0, bytearray(true_lock_ack))
        await _spin_until(lambda: progress.acknowledged)
        # Only its own bb0b op-response completes the wait.
        session._notify(0, bytearray(genuine_lock_result))

    feeder = asyncio.create_task(feed())
    op_task = asyncio.create_task(
        session.execute_operation(
            command,
            "force_lock",
            ack_matcher=_ack_matcher(Commands.LOCK, 0x00),
            response_matcher=_operation_response_matcher(Commands.LOCK),
            response_timeout=5.0,
            progress=progress,
        )
    )
    result = await op_task
    await feeder

    assert result == genuine_lock_result
    # It went through the real ack stage; the stale bb0a did not short-circuit.
    assert progress.acknowledged is True
    # The stale op-response and the settle stayed on the state-callback path.
    assert stale_unlock_result in seen
    assert settled_unlocked in seen


@pytest.mark.asyncio
async def test_replay_incident2_settled_bb02_never_completes_force_unlock() -> None:
    """Incident 2 (substitute): a settled bb02 push never completes an operation.

    The documented ~6.4 s stale bb02 was not present in the log; this replays
    the closest same-family case -- a settled STATUS-UNLOCKED push (bb02...03)
    that the old wait claimed for a force_unlock. A GETSTATUS status push
    (0xBB 0x02) is not an operation frame, so it completes the wait at NEITHER
    stage; only the unlock's own bb0a does.
    """
    seen: list[bytes] = []
    session, _ = _make_session(seen.append)
    progress = OperationProgress()
    command = session.build_command(Commands.UNLOCK)  # force_unlock: opcode 0x0a

    # Verbatim field frames (byte[3] checksums valid as captured).
    settled_unlocked = bytes.fromhex("bb02003e0200000003000000000000000000")
    true_unlock_ack = bytes.fromhex("aa0a004a0000000000000000000000000200")
    genuine_unlock_result = bytes.fromhex("bb0a003b0000000000000000000000000000")

    op_task: asyncio.Task[bytes] | None = None

    async def feed() -> None:
        await _spin_until(lambda: progress.command_written)
        # A settled status push during the ack stage does not complete it.
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert op_task is not None and not op_task.done()
        session._notify(0, bytearray(true_unlock_ack))
        await _spin_until(lambda: progress.acknowledged)
        # A settled status push during the op-response stage does not either.
        session._notify(0, bytearray(settled_unlocked))
        await asyncio.sleep(0)
        assert not op_task.done()
        session._notify(0, bytearray(genuine_unlock_result))

    feeder = asyncio.create_task(feed())
    op_task = asyncio.create_task(
        session.execute_operation(
            command,
            "force_unlock",
            ack_matcher=_ack_matcher(Commands.UNLOCK, 0x00),
            response_matcher=_operation_response_matcher(Commands.UNLOCK),
            response_timeout=5.0,
            progress=progress,
        )
    )
    result = await op_task
    await feeder

    assert result == genuine_unlock_result
    assert progress.acknowledged is True
    # Both settled pushes stayed on the state-callback path, never claimed.
    assert seen.count(settled_unlocked) == 2

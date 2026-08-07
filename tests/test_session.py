"""Session-level tests for frame admission and the solicited-response matcher."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from yalexs_ble import util
from yalexs_ble.const import Commands, SettingType
from yalexs_ble.lock import _settings_response_matcher
from yalexs_ble.session import RESPONSE_FRAME_LEN, ResponseError, Session

# Verbatim field frames (2026-07-16 capture, YUR/DEL fw 2.1.0): the READSETTING
# acknowledgment and, ~40 ms later, the 0xBB frame carrying the stored value
# (Timed 1800: both uint16 timers set to 1800).
READ_ACK = bytes.fromhex("aa0400282800000000000000000000000200")
READ_ANSWER = bytes.fromhex("bb0400fb2800000008070807000000000000")
# The same value frame with its checksum byte cleared, so _validate_response
# rejects it the way a corrupted frame off the air is rejected.
CORRUPT_ANSWER = bytes.fromhex("bb0400002800000008070807000000000000")


def _make_session(received: list[bytes]) -> Session:
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
    session = _make_session(received)
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
    session = _make_session(received)

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
    previous command's matcher still in place.
    """
    received: list[bytes] = []
    session = _make_session(received)
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
    # The corrupt frame was withheld from the state callback; only the valid
    # one reached it, and only it answered.
    assert received == [READ_ANSWER]


@pytest.mark.asyncio
async def test_corrupt_frame_on_every_attempt_raises_and_leaves_the_slot_empty() -> (
    None
):
    """Three corrupt frames exhaust the retries and the error reaches the caller."""
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(CORRUPT_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    with pytest.raises(ResponseError):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session.client.write_gatt_char.await_count == 3
    assert session._notify_future is None
    assert session._notify_matcher is None


def _short_timeout(_seconds: float) -> object:
    """Replacement for util.asyncio_timeout that expires almost immediately."""
    return asyncio.timeout(0.01)


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_success() -> None:
    """On the happy path the notify slot is empty when _locked_write returns."""
    received: list[bytes] = []
    session = _make_session(received)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    result = await session._locked_write(bytearray(18), "auto_lock_status")

    assert result == READ_ANSWER
    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_locked_write_clears_slot_on_timeout() -> None:
    """When no response arrives, the timeout path disarms the notify slot."""
    received: list[bytes] = []
    session = _make_session(received)
    # The write succeeds but no notify is ever delivered.
    session.client.write_gatt_char = AsyncMock()

    with (
        patch.object(util, "asyncio_timeout", _short_timeout),
        pytest.raises(TimeoutError),
    ):
        await session._locked_write(bytearray(18), "auto_lock_status")

    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_locked_write_clears_slot_when_the_write_errors() -> None:
    """A write that raises disarms the notify slot on the way out.

    Both slots are armed before the GATT write, so a write error exits with
    this attempt's own future still in them. Only a timeout used to disarm.
    """
    received: list[bytes] = []
    session = _make_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )
    session.client.write_gatt_char = AsyncMock(side_effect=BleakError("disconnected"))

    with pytest.raises(BleakError):
        await session._locked_write(bytearray(18), "auto_lock_status", matcher)

    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_locked_write_clears_slot_when_the_wait_is_cancelled() -> None:
    """Cancelling the waiting task disarms the notify slot on the way out.

    A cancellation from outside (a caller timeout, a shutdown) leaves the
    future cancelled; a frame arriving before the next command re-arms the
    slot would land on it. Only a timeout used to disarm.
    """
    received: list[bytes] = []
    session = _make_session(received)
    matcher = _settings_response_matcher(
        Commands.READSETTING.value, SettingType.AUTOLOCK.value
    )
    written = asyncio.Event()

    async def deliver(*_args: object, **_kwargs: object) -> None:
        written.set()

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    task = asyncio.create_task(
        session._locked_write(bytearray(18), "auto_lock_status", matcher)
    )
    await written.wait()
    armed = session._notify_future
    assert armed is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert session._notify_future is None
    assert session._notify_matcher is None


@pytest.mark.asyncio
async def test_a_frame_delivered_after_the_wait_ended_is_dropped() -> None:
    """A frame that arrives once the wait has ended does not resolve it.

    A timeout ends the wait by cancelling the task, which cancels the future
    it is suspended on, and the task disarms the slot only when it resumes on
    the next pass of the loop. A notification dispatched in between finds the
    slot armed and the future already done, and answering it there would raise
    InvalidStateError into the notify dispatcher. Cancelling the task directly
    puts the slot in that state.
    """
    received: list[bytes] = []
    session = _make_session(received)
    written = asyncio.Event()

    async def deliver(*_args: object, **_kwargs: object) -> None:
        written.set()

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    task = asyncio.create_task(session._locked_write(bytearray(18), "auto_lock_status"))
    await written.wait()
    armed = session._notify_future
    assert armed is not None
    task.cancel()
    assert armed.cancelled()

    session._notify(0, bytearray(READ_ANSWER))

    with pytest.raises(asyncio.CancelledError):
        await task
    # The frame still reached the state callback; only the answer was dropped.
    assert received == [READ_ANSWER]
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_a_corrupt_frame_delivered_after_the_wait_ended_is_dropped() -> None:
    """The error path drops its frame in that window too.

    Same state as the test above, reached by a frame that fails validation:
    the error has no wait left to carry it.
    """
    received: list[bytes] = []
    session = _make_session(received)
    written = asyncio.Event()

    async def deliver(*_args: object, **_kwargs: object) -> None:
        written.set()

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)
    task = asyncio.create_task(session._locked_write(bytearray(18), "auto_lock_status"))
    await written.wait()
    armed = session._notify_future
    assert armed is not None
    task.cancel()
    assert armed.cancelled()

    session._notify(0, bytearray(CORRUPT_ANSWER))

    with pytest.raises(asyncio.CancelledError):
        await task
    # Admission withholds the frame from the state callback, so nothing is
    # published; what this pins is that the error finds no wait to land on.
    assert received == []
    assert session._notify_future is None


@pytest.mark.asyncio
async def test_late_notify_after_timeout_is_ignored() -> None:
    """A frame arriving after the timeout cleared the slot is a no-op.

    The timeout leaves ``_notify_future`` at None, so a late frame reaches the
    state callback but resolves no wait and raises nothing.
    """
    received: list[bytes] = []
    session = _make_session(received)
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
    session = _make_session(received)
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


SESSION_KEY = bytes(range(16))


def _make_encrypted_session(received: list[bytes]) -> Session:
    """Create a Session with the real CBC contexts rather than pass-through."""
    client = MagicMock(is_connected=True)
    session = Session(client, "testlock", asyncio.Lock(), set(), received.append)
    session.set_key(SESSION_KEY)
    return session


def _lock_side_frames(*plaintexts: bytes) -> list[bytes]:
    """Encrypt frames the way the lock does, as one chained stream.

    The first 16 bytes are one CBC block and the last two travel in the clear,
    and the context chains across frames, which is why a frame that reaches
    our decryptor out of step desynchronizes every frame after it.
    """
    encryptor = Cipher(algorithms.AES(SESSION_KEY), modes.CBC(bytes(0x10))).encryptor()
    return [encryptor.update(p[0:0x10]) + p[0x10:0x12] for p in plaintexts]


@pytest.mark.asyncio
async def test_a_short_frame_is_dropped_before_it_can_poison_the_cipher() -> None:
    """A wrong-length payload never reaches the cipher, so the stream survives.

    The CBC context consumes ciphertext in 16-byte blocks. A partial block
    would stay buffered inside it and every later frame on the connection
    would decrypt misaligned, which only a reconnect's set_key could undo. The
    frame after the runt must still decode.
    """
    received: list[bytes] = []
    session = _make_encrypted_session(received)
    first, second = _lock_side_frames(READ_ACK, READ_ANSWER)

    session._notify(0, bytearray(first))
    session._notify(0, bytearray(b"\x00" * 10))
    session._notify(0, bytearray(second))

    assert received == [READ_ACK, READ_ANSWER]


@pytest.mark.asyncio
async def test_an_unsolicited_invalid_frame_is_withheld_from_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame arriving with no wait armed is validated too, and dropped.

    The state callback drives the consumer's state whether or not a command is
    outstanding, so an unsolicited frame has to clear the same bar.
    """
    received: list[bytes] = []
    session = _make_session(received)

    with caplog.at_level(logging.INFO, logger="yalexs_ble.session"):
        session._notify(0, bytearray(CORRUPT_ANSWER))
        session._notify(0, bytearray(len(CORRUPT_ANSWER) // 2))

    assert received == []
    assert "dropping invalid frame" in caplog.text
    # The rejection reaches the user through _locked_write's exhausted
    # retries, so it names the length it wanted as well as the one it got.
    assert "9 bytes is not an 18-byte response frame" in caplog.text


@pytest.mark.asyncio
async def test_the_frame_length_gate_is_strict_equality() -> None:
    """A longer payload is refused as well as a shorter one.

    A frame carrying more than the response length would otherwise validate on
    its first 18 decrypted bytes and be passed on with a tail that was never
    decrypted.
    """
    received: list[bytes] = []
    session = _make_session(received)

    session._notify(0, bytearray(READ_ANSWER + READ_ANSWER))

    assert len(READ_ANSWER) == RESPONSE_FRAME_LEN
    assert received == []


@pytest.mark.asyncio
async def test_a_session_without_a_state_callback_still_answers_its_wait() -> None:
    """The secure session carries no state callback, and admission assumes none.

    Its handshake frames go through the same length gate and checksum as every
    other frame, and must still resolve the wait that asked for them.
    """
    client = MagicMock(is_connected=True)
    session = Session(client, "testlock", asyncio.Lock(), set())
    session.decrypt = bytes  # type: ignore[method-assign, assignment]
    session.cipher_encrypt = MagicMock(update=bytes)

    async def deliver(*_args: object, **_kwargs: object) -> None:
        session._notify(0, bytearray(READ_ANSWER))

    session.client.write_gatt_char = AsyncMock(side_effect=deliver)

    assert await session.execute(bytearray(18), "auto_lock_status") == READ_ANSWER

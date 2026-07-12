import asyncio
import contextlib
from collections.abc import Callable, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bleak.exc import BleakError
from bleak_retry_connector import BLEDevice
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from yalexs_ble.const import (
    FIRMWARE_REVISION_CHARACTERISTIC,
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    VALUE_TO_LOCK_STATUS,
    AutoLockMode,
    AutoLockState,
    Commands,
    LockInfo,
    LockOperationRemoteType,
    LockOperationSource,
    LockStateValue,
    LockStatus,
    OperationError,
    SettingType,
)
from yalexs_ble.lock import (
    AA_BATTERY_VOLTAGE_TO_PERCENTAGE,
    OPERATION_RESPONSE_TIMEOUT,
    SECUREMODE_OPERATION_BYTE,
    UNLATCH_OPERATION_BYTE,
    UNLATCH_OPERATION_RESPONSE_TIMEOUT,
    Lock,
    _ack_matcher,
    _operation_response_matcher,
    _settings_response_matcher,
    convert_voltage_to_percentage,
)
from yalexs_ble.session import (
    DisconnectedError,
    OperationIncompleteError,
    OperationProgress,
    Session,
    UnlatchError,
)
from yalexs_ble.util import _simple_checksum


def test_aa_battery_voltage_to_percentage_is_monotonic() -> None:
    """Percentage must be non-increasing as voltage decreases.

    Guards against copy/paste regressions in the lookup table — a non-monotonic
    table makes ``convert_voltage_to_percentage`` return higher percentages for
    lower voltages, which erodes user trust in the battery indicator.
    """
    sorted_pairs = sorted(AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    percents = [pct for _, pct in sorted_pairs]
    assert percents == sorted(percents), (
        f"voltage→pct table is non-monotonic: {sorted_pairs}"
    )


def test_convert_voltage_to_percentage_is_monotonic_across_table() -> None:
    """``convert_voltage_to_percentage`` must be non-decreasing in voltage."""
    voltages = sorted(v for v, _ in AA_BATTERY_VOLTAGE_TO_PERCENTAGE)
    results = [convert_voltage_to_percentage(v) for v in voltages]
    assert results == sorted(results), (
        f"convert_voltage_to_percentage is non-monotonic across table voltages: "
        f"{list(zip(voltages, results, strict=True))}"
    )


def test_create_lock() -> None:
    Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )


@pytest.mark.asyncio
async def test_connection_canceled_on_disconnect() -> None:
    disconnect_mock = AsyncMock()
    mock_client = MagicMock(connected=True, disconnect=disconnect_mock)
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", delegate=""),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )
    lock.client = mock_client

    async def connect_and_wait() -> None:
        await lock.connect()
        await asyncio.sleep(2)

    with patch("yalexs_ble.lock.Lock.connect"):
        task = asyncio.create_task(connect_and_wait())
        await asyncio.sleep(0)
        task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert task.cancelled() is True


def test_parse_operation_source() -> None:
    """Test parsing operation source and remote type."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )

    # Test remote source with BLE type
    source, remote_type = lock._parse_operation_source(0x00, 0x03)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.BLE

    # Test manual source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x01, 0x03)
    assert source is LockOperationSource.MANUAL
    assert remote_type is None

    # Test auto lock source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x05, 0x00)
    assert source is LockOperationSource.AUTO_LOCK
    assert remote_type is None

    # Test PIN source (remote_type should be None)
    source, remote_type = lock._parse_operation_source(0x0B, 0x03)
    assert source is LockOperationSource.PIN
    assert remote_type is None

    # Test unknown source
    source, remote_type = lock._parse_operation_source(0x99, 0x03)
    assert source is LockOperationSource.UNKNOWN
    assert remote_type is None

    # Test remote source with unknown remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x99)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN

    # Test remote source with UNKNOWN (0x00) remote type
    source, remote_type = lock._parse_operation_source(0x00, 0x00)
    assert source is LockOperationSource.REMOTE
    assert remote_type is LockOperationRemoteType.UNKNOWN


def test_parse_lock_command_response_jammed() -> None:
    """LOCK op-response with a MECH_* result (byte[15]) parses as JAMMED."""
    lock = _make_lock()

    # Real lock-jam capture: byte[15] = 0x1F MECH_POSITION. byte[3] (0x1B
    # here) is only the frame checksum, not a status.
    frame = bytes.fromhex("bb0b001b00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_unlock_command_response_jammed() -> None:
    """UNLOCK op-response with a MECH_* result (byte[15]) parses as JAMMED.

    The old byte[3] path missed this: an unlock jam's checksum is 0x1C, not
    the 0x1B it looked for. The result is in byte[15] (0x1F MECH_POSITION)
    regardless of direction.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a001c00000000000000000000001f0000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_lock_command_response_success_is_no_update() -> None:
    """A successful LOCK op-response (byte[15]=0x00) carries no state update.

    The op-response reports the result of the issued command; which state
    resulted is known to the command issuer, not the parser (lock and
    securemode op-responses are byte-identical), so the parser emits nothing.
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0b003a0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_unlock_command_response_success_is_no_update() -> None:
    """A successful UNLOCK op-response (byte[15]=0x00) carries no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0a003b0000000000000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == []


def test_parse_getstatus_staticposition() -> None:
    """A settled GETSTATUS lock state of 0x07 (STATICPOSITION) parses as JAMMED."""
    lock = _make_lock()

    # bb02 GETSTATUS, byte[4]=0x02 LOCK_ONLY, byte[8]=0x07 (settled jam state).
    frame = bytes.fromhex("bb02003a0200000007000000000000000000")
    result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]


def test_parse_success_op_response_with_0200_trailer_is_no_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The issue #317 fixture: byte[3] shifts with the plaintext trailer.

    A successful unlock op-response with the ``0200`` CommandType trailer
    moves byte[3] to 0x39. Keying off byte[3] would miss it; keying off
    byte[15]=0x00 recognizes it as a successful op-response with no state
    update -- and it must not log "Unknown state".
    """
    lock = _make_lock()

    frame = bytes.fromhex("bb0a00390000000000000000000000000200")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def test_parse_lock_activity_is_no_update(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A LOCK_ACTIVITY (0xBB 0x2D) frame is recognized with no state update."""
    lock = _make_lock()

    frame = bytes.fromhex("bb2d008000000000000000000000000000")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)
        lock._internal_state_callback(frame)

    assert result is not None
    assert list(result) == []
    assert "Unknown state" not in caplog.text


def test_parse_non_mech_error_is_jammed_and_logs_decoded_name(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-MECH failure result still parses as JAMMED and logs its name."""
    lock = _make_lock()

    # byte[15] = 0x32 VBAT_LOW (synthetic; no real capture for a non-MECH error).
    # Captured at WARNING: an operation failure must be visible at default
    # log levels, not only in a debug session.
    frame = bytes.fromhex("bb0b00000000000000000000000000320000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert "0x32" in caplog.text
    assert "VBAT_LOW" in caplog.text


def test_parse_unknown_error_code_is_jammed_and_logs_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmapped non-zero result is JAMMED and logs the raw value as unknown."""
    lock = _make_lock()

    frame = bytes.fromhex("bb0b00000000000000000000000000770000")
    with caplog.at_level("WARNING", logger="yalexs_ble.lock"):
        result = lock._parse_state(frame)

    assert result is not None
    assert list(result) == [LockStatus.JAMMED]
    assert "0x77" in caplog.text
    assert "unknown" in caplog.text


def test_last_op_error_is_retained() -> None:
    """The op-response result byte[15] is retained on the lock instance."""
    # Collected and compared once: asserting on the attribute per step narrows
    # it (mypy keeps the narrowing across the _parse_state call) and the later
    # steps are then flagged unreachable.
    lock = _make_lock()
    seen: list[int | None] = [lock._last_op_error]

    lock._parse_state(bytes.fromhex("bb0b001b00000000000000000000001f0000"))
    seen.append(lock._last_op_error)

    lock._parse_state(bytes.fromhex("bb0b003a0000000000000000000000000000"))
    seen.append(lock._last_op_error)

    assert seen == [None, 0x1F, 0x00]


def test_parse_bogus_frame_is_none_and_logs_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frame with an unrecognized flag byte is not recognized and still logs."""
    lock = _make_lock()

    frame = bytes.fromhex("cc00000000000000000000000000000000")
    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        assert lock._parse_state(frame) is None
        lock._internal_state_callback(frame)

    assert "Unknown state" in caplog.text


def test_parse_operation_ack_reports_no_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operation acks (0xAA LOCK/UNLOCK) are recognized but carry no state.

    They echo the request type with no result, so a securemode request (the
    0x0B Lock echo) used to display a false LOCKED. State now comes from the
    op-response; the ack is recognized (empty iterable), emits nothing, and
    must not surface as an unknown frame.
    """
    states: list[list[LockStateValue]] = []
    lock = _make_lock(lambda s: states.append(list(s)))

    with caplog.at_level("INFO", logger="yalexs_ble.lock"):
        for frame_hex in (
            "aa0b00490000000000000000000000000200",
            "aa0a004a0000000000000000000000000200",
        ):
            frame = bytes.fromhex(frame_hex)
            result = lock._parse_state(frame)
            assert result is not None
            assert list(result) == []
            lock._internal_state_callback(frame)

    assert states == []  # the state callback was never invoked
    assert "Unknown state" not in caplog.text


def test_ack_and_op_response_callbacks_fire() -> None:
    """ack_callback fires per operation ack; op_response_callback per op-response.

    The op-response hook fires for both a success and a failure result,
    independent of whatever state the parse emits.
    """
    ack_calls: list[int] = []
    op_calls: list[int] = []
    lock = _make_lock(
        ack_callback=lambda: ack_calls.append(1),
        op_response_callback=lambda: op_calls.append(1),
    )

    # Each operation ack fires ack_callback exactly once and emits no state.
    lock_ack = lock._parse_state(bytes.fromhex("aa0b00490000000000000000000000000200"))
    unlock_ack = lock._parse_state(
        bytes.fromhex("aa0a004a0000000000000000000000000200")
    )

    assert lock_ack is not None
    assert unlock_ack is not None
    assert list(lock_ack) == []
    assert list(unlock_ack) == []
    assert ack_calls == [1, 1]
    assert op_calls == []  # acks are not op-responses

    # A success op-response (byte[15]=0x00) and a failure one (byte[15]=0x1F)
    # both stamp op_response_callback, regardless of the emitted state.
    success = lock._parse_state(bytes.fromhex("bb0a003b0000000000000000000000000000"))
    failure = lock._parse_state(bytes.fromhex("bb0a001c00000000000000000000001f0000"))

    assert op_calls == [1, 1]
    assert success is not None
    assert failure is not None
    assert list(success) == []
    assert list(failure) == [LockStatus.JAMMED]
    # The op-responses did not spuriously fire the ack hook.
    assert ack_calls == [1, 1]


def test_internal_state_callback_emits_recognized_state() -> None:
    """A recognized frame with state content reaches the state callback."""
    received: list[list[LockStateValue]] = []
    lock = _make_lock(lambda states: received.append(list(states)))

    # Settled status push after a jam: GETSTATUS/LOCK_ONLY with state 0x07
    # (production capture).
    lock._internal_state_callback(bytes.fromhex("bb02003a0200000007000000000000000000"))

    assert received == [[LockStatus.JAMMED]]


def test_jammed_maps_to_the_settled_static_position_value() -> None:
    """JAMMED is the settled post-jam status value 0x07 (STATICPOSITION)."""
    assert LockStatus(0x07) is LockStatus.JAMMED
    assert VALUE_TO_LOCK_STATUS[0x07] is LockStatus.JAMMED


def test_ack_matcher_matches_only_the_written_operation() -> None:
    """The ack matcher keys on 0xAA + the written opcode + operation byte."""
    matches = _ack_matcher(0x0B, 0x04)

    # Correct ack: 0xAA, opcode 0x0B, operation byte 0x04.
    assert matches(bytes.fromhex("aa0b00450400000000000000000000000200"))
    # Same opcode but operation byte 0x00 -- a plain-lock ack, not securemode.
    assert not matches(bytes.fromhex("aa0b00490000000000000000000000000200"))
    # Wrong opcode (0x0A).
    assert not matches(bytes.fromhex("aa0a004a0000000000000000000000000200"))
    # An op-response (0xBB), not an acknowledgement.
    assert not matches(bytes.fromhex("bb0b00450400000000000000000000000200"))


def test_operation_response_matcher_matches_only_its_opcode() -> None:
    """The op-response matcher keys on 0xBB + the sent opcode, full length."""
    matches = _operation_response_matcher(0x0A)

    # An 18-byte 0xBB 0x0A op-response.
    assert matches(bytes.fromhex("bb0a00000000000000000000000000000200"))
    # Wrong opcode (0x0B).
    assert not matches(bytes.fromhex("bb0b00000000000000000000000000000200"))
    # An acknowledgement (0xAA), not an op-response.
    assert not matches(bytes.fromhex("aa0a00000000000000000000000000000200"))
    # Truncated: byte[15] (the result) is not present.
    assert not matches(bytes.fromhex("bb0a0000000000000000"))


def _make_lock(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
    *,
    ack_callback: Callable[[], None] | None = None,
    op_response_callback: Callable[[], None] | None = None,
) -> Lock:
    return Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock"),
        "0800200c9a66",
        1,
        "mylock",
        state_callback,
        ack_callback=ack_callback,
        op_response_callback=op_response_callback,
    )


def test_parse_auto_lock_state_timed_from_wire() -> None:
    """Real capture: both uint16 timers set to 1800 -> Timed 30 min.

    Front Door READSETTING response, YUR/DEL fw 2.1.0 (2026-07-05 capture).
    """
    lock = _make_lock()
    response = bytes.fromhex("bb0400fb2800000008070807000000000000")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 1800)


def test_parse_auto_lock_state_off_from_wire() -> None:
    """Real capture: all-zero setting value -> auto-lock off.

    Back Door READSETTING response (2026-07-05 capture).
    """
    lock = _make_lock()
    response = bytes.fromhex("bb0400192800000000000000000000000000")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.OFF, 0)


def test_parse_auto_lock_state_old_encoding_reads_user_value() -> None:
    """A value written by a release before the two-timer encoding -> Timed 30.

    Earlier releases stored the user's seconds in the never-opened timer and a
    fixed 90 in the door-close timer, so Timed(30) was written as 1e 00 5a 00.
    The decode reports the never-opened timer, so the value reads back as set.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("1e005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 30)


def test_parse_auto_lock_state_zero_never_opened_falls_back() -> None:
    """A zero never-opened timer falls back to the door-close timer.

    Synthetic value exercising the branch; not a captured device value.
    """
    lock = _make_lock()
    response = bytes(8) + bytes.fromhex("00005a00")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.TIMER, 90)


def test_parse_auto_lock_state_instant_never_opened_only() -> None:
    """Derivation branch: never-opened timer set, door-close timer zero -> Instant.

    Synthetic value exercising the branch; not a captured device value.
    """
    lock = _make_lock()
    response = bytes(8) + (0x0005).to_bytes(4, "little")
    result = lock._parse_auto_lock_state(response)
    assert result == AutoLockState(AutoLockMode.INSTANT, 5)


class _CommandCaptureSession:
    """Minimal Session stand-in that captures executed commands.

    build_operation_command mirrors Session's 18-byte frame layout
    (EE, opcode, cmd byte at [4], ClearText trailer marker at [16]).
    """

    def __init__(self) -> None:
        self.sent: list[bytearray] = []

    def build_operation_command(self, opcode: int, cmd_byte: int) -> bytearray:
        cmd = bytearray(0x12)
        cmd[0x00] = 0xEE
        cmd[0x01] = opcode
        cmd[0x04] = cmd_byte
        cmd[0x10] = 0x02
        return cmd

    async def execute(
        self,
        command: bytearray,
        command_name: str,
        response_matcher: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        self.sent.append(command)
        return b""


async def _set_auto_lock_payload(mode: AutoLockMode, duration: int) -> bytearray:
    """Run set_auto_lock against a capture session; return the sent command."""
    lock = _make_lock()
    session = _CommandCaptureSession()
    lock.session = session  # type: ignore[assignment]
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    await lock.set_auto_lock(mode, duration)
    assert len(session.sent) == 1
    return session.sent[0]


@pytest.mark.asyncio
async def test_set_auto_lock_timed_encodes_seconds_in_both_timers() -> None:
    """Timed(1800) -> both uint16 timers = 1800 -> [8:12] = 08 07 08 07."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 1800)
    assert cmd[0x01] == Commands.WRITESETTING.value
    assert cmd[0x04] == 0x28  # auto-lock setting id
    assert cmd[0x08:0x0C] == bytes.fromhex("08070807")


@pytest.mark.asyncio
async def test_set_auto_lock_instant_encodes_never_opened_only() -> None:
    """Instant(5) -> never-opened timer = 5, door-close 0 -> [8:12] = 05 00 00 00."""
    cmd = await _set_auto_lock_payload(AutoLockMode.INSTANT, 5)
    assert cmd[0x08:0x0C] == bytes.fromhex("05000000")


@pytest.mark.asyncio
async def test_set_auto_lock_off_encodes_zero() -> None:
    """Off -> value = 0 regardless of the duration argument."""
    cmd = await _set_auto_lock_payload(AutoLockMode.OFF, 1800)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_set_auto_lock_duration_out_of_range_raises() -> None:
    """Durations must fit a uint16 timer; 0xFFFF+ is rejected (app rule 1-65534)."""
    with pytest.raises(ValueError, match="out of range"):
        await _set_auto_lock_payload(AutoLockMode.TIMER, 0xFFFF)


@pytest.mark.asyncio
async def test_set_auto_lock_round_trips_through_decode() -> None:
    """A value we write, echoed back by the lock, decodes to what we set."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 1800)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(
        AutoLockMode.TIMER, 1800
    )


@pytest.mark.asyncio
async def test_set_auto_lock_timed_accepts_upper_bound() -> None:
    """Timed(0xFFFE) is the largest accepted duration.

    Both uint16 timers take the seconds, so [8:12] = fe ff fe ff.
    """
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0xFFFE)
    assert cmd[0x08:0x0C] == bytes.fromhex("fefffeff")


@pytest.mark.asyncio
async def test_set_auto_lock_timed_zero_duration_encodes_off_shape() -> None:
    """Timed with a zero duration collapses to the off shape: an all-zero value."""
    cmd = await _set_auto_lock_payload(AutoLockMode.TIMER, 0)
    assert cmd[0x08:0x0C] == bytes(4)


@pytest.mark.asyncio
async def test_auto_lock_status_issues_read() -> None:
    """auto_lock_status sends a READSETTING for the auto-lock setting.

    The wait completes on the acknowledgment, which carries no value, so the
    method returns nothing; the stored setting arrives later as a settings
    response on the notify path.
    """
    lock = _make_lock()
    session = _CommandCaptureSession()
    lock.session = session  # type: ignore[assignment]
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)
    await lock.auto_lock_status()
    assert len(session.sent) == 1
    assert session.sent[0][0x01] == Commands.READSETTING.value
    assert session.sent[0][0x04] == SettingType.AUTOLOCK.value


@pytest.mark.asyncio
async def test_set_auto_lock_instant_round_trips_through_decode() -> None:
    """Instant(5), encoded then decoded, returns Instant(5)."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.INSTANT, 5)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(AutoLockMode.INSTANT, 5)


@pytest.mark.asyncio
async def test_set_auto_lock_off_round_trips_through_decode() -> None:
    """Off, encoded then decoded, returns Off with a zero duration."""
    lock = _make_lock()
    cmd = await _set_auto_lock_payload(AutoLockMode.OFF, 0)
    echoed = bytes([0xBB, 0x04, 0x00, 0x00, 0x28, 0, 0, 0]) + bytes(cmd[0x08:0x0C])
    assert lock._parse_auto_lock_state(echoed) == AutoLockState(AutoLockMode.OFF, 0)


def test_parse_state_readsetting_ack_ignored() -> None:
    """The READSETTING (0x04) transport ACK carries no state -> recognized, ignored.

    Real ACK frame for an auto-lock READSETTING (2026-07-05 capture); must return
    an empty iterable (not None), so it is never logged as an unknown frame.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa0400282800000000000000000000000200")
    assert lock._parse_state(ack) == ()


def test_parse_state_writesetting_ack_ignored() -> None:
    """The WRITESETTING (0x03) transport ACK carries no state -> recognized, ignored.

    Real ACK frame for an auto-lock write of Timed(90) (2026-07-16 capture); the
    stored value is echoed at [8:12] but the frame is only the acknowledgment --
    the authoritative value is the 0xBB settings response that follows.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa030075280000005a005a00000000000200")
    assert lock._parse_state(ack) == ()


def test_parse_state_ack_for_other_opcode_is_unknown() -> None:
    """ACK recognition is scoped to the settings opcodes.

    An 0xAA frame whose opcode is neither a lock/unlock ack nor a settings ack
    is not claimed: it falls through to None, so a new acknowledgment type on
    another model still surfaces as an unknown frame instead of being silently
    dropped. Frame built from the READSETTING ACK capture above with the opcode
    byte changed to LOCK_ACTIVITY (0x2D), which is recognized on the 0xBB flag
    only.
    """
    lock = _make_lock()
    ack = bytes.fromhex("aa2d00282800000000000000000000000200")
    assert lock._parse_state(ack) is None


def test_settings_response_matcher_takes_value_frame_not_ack() -> None:
    """The matcher keys on 0xBB + the settings opcode + the setting id.

    All frames verbatim from the 2026-07-16 field capture: a settings command
    is answered by an 0xAA acknowledgment ~40 ms before the 0xBB value frame,
    and the acknowledgment's zero value field decodes as auto-lock off.
    """
    write_matcher = _settings_response_matcher(
        Commands.WRITESETTING.value, SettingType.AUTOLOCK.value
    )

    read_response = bytes.fromhex("bb0400fb2800000008070807000000000000")
    write_ack = bytes.fromhex("aa030075280000005a005a00000000000200")
    write_response = bytes.fromhex("bb030066280000005a005a00000000000000")
    battery_answer = bytes.fromhex("bb0200a50f00000079140000000000000200")

    assert write_matcher(write_response)
    assert not write_matcher(write_ack)
    assert not write_matcher(read_response)  # wrong opcode for the write
    assert not write_matcher(battery_answer)
    assert not write_matcher(write_response[:4])  # truncated below the setting id


_CHAR_DATA: dict[str, bytes] = {
    MODEL_NUMBER_CHARACTERISTIC: b"ASL-03",
    SERIAL_NUMBER_CHARACTERISTIC: b"12345",
    FIRMWARE_REVISION_CHARACTERISTIC: b"2.0.0",
}

# Model is read first, then serial, firmware.
_CHAR_ORDER: tuple[str, ...] = (
    MODEL_NUMBER_CHARACTERISTIC,
    SERIAL_NUMBER_CHARACTERISTIC,
    FIRMWARE_REVISION_CHARACTERISTIC,
)


def _make_lock_with_mock_client(
    side_effects: dict[str, Exception] | None = None,
) -> tuple[Lock, MagicMock]:
    """Create a Lock with a mock BLE client for lock_info tests."""
    lock = Lock(
        lambda: BLEDevice("aa:bb:cc:dd:ee:ff", "lock", details=None),
        "0800200c9a66",
        1,
        "mylock",
        lambda _: None,
    )
    mock_client = MagicMock()
    mock_client.is_connected = True
    lock.client = mock_client
    lock.session = MagicMock()
    lock.secure_session = MagicMock()

    effects = side_effects or {}

    # Map each characteristic UUID to a unique mock object so
    # read_gatt_char can identify which UUID is being read.
    char_mocks: dict[str, MagicMock] = {}
    mock_to_uuid: dict[int, str] = {}
    for uuid in _CHAR_ORDER:
        m = MagicMock()
        char_mocks[uuid] = m
        mock_to_uuid[id(m)] = uuid

    mock_client.services.get_characteristic = char_mocks.get

    async def read_gatt_char(char: MagicMock) -> bytes:
        uuid = mock_to_uuid[id(char)]
        if uuid in effects:
            raise effects[uuid]
        return _CHAR_DATA[uuid]

    mock_client.read_gatt_char = read_gatt_char
    mock_client._mock_to_uuid = mock_to_uuid
    return lock, mock_client


@pytest.mark.asyncio
async def test_lock_info_success() -> None:
    """Test lock_info reads all characteristics successfully."""
    lock, _ = _make_lock_with_mock_client()

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="ASL-03",
        serial="12345",
        firmware="2.0.0",
    )


@pytest.mark.asyncio
async def test_lock_info_partial_failure() -> None:
    """Test lock_info continues when individual reads fail."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={SERIAL_NUMBER_CHARACTERISTIC: BleakError("Connection dropped")}
    )

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_all_reads_fail() -> None:
    """Test lock_info returns all Unknown when every read fails."""
    lock, _ = _make_lock_with_mock_client(
        side_effects={uuid: BleakError("Failed") for uuid in _CHAR_ORDER}
    )

    info = await lock.lock_info()

    assert info == LockInfo(
        manufacturer="Yale/August",
        model="",
        serial="aa:bb:cc:dd:ee:ff",
        firmware="Unknown",
    )


@pytest.mark.asyncio
async def test_lock_info_timeout() -> None:
    """Test lock_info returns partial results when reads hang."""
    lock, mock_client = _make_lock_with_mock_client()

    async def hang_forever(char: MagicMock) -> bytes:
        await asyncio.sleep(999)
        return b""  # unreachable

    mock_client.read_gatt_char = hang_forever

    with patch("yalexs_ble.lock.LOCK_INFO_TIMEOUT", 0):
        info = await lock.lock_info()

    # All reads hung so no results, but we get defaults instead of an exception
    assert info.manufacturer == "Yale/August"
    assert info.model == ""
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "Unknown"


@pytest.mark.asyncio
async def test_lock_info_missing_characteristic() -> None:
    """Test lock_info skips missing characteristics instead of aborting."""
    lock, mock_client = _make_lock_with_mock_client()

    original_get = mock_client.services.get_characteristic

    def get_char_skip_serial(uuid: str) -> MagicMock | None:
        if uuid == SERIAL_NUMBER_CHARACTERISTIC:
            return None
        return original_get(uuid)

    mock_client.services.get_characteristic = get_char_skip_serial

    info = await lock.lock_info()

    assert info.manufacturer == "Yale/August"
    assert info.model == "ASL-03"
    assert info.serial == "aa:bb:cc:dd:ee:ff"
    assert info.firmware == "2.0.0"


@pytest.mark.asyncio
async def test_lock_info_reads_model_first() -> None:
    """Test that model is read first so it's available as early as possible."""
    lock, mock_client = _make_lock_with_mock_client()
    call_order: list[str] = []
    original_read = mock_client.read_gatt_char
    mock_to_uuid = mock_client._mock_to_uuid

    async def tracking_read(char: MagicMock) -> bytes:
        call_order.append(mock_to_uuid[id(char)])
        return await original_read(char)

    mock_client.read_gatt_char = tracking_read

    await lock.lock_info()

    assert call_order[0] == MODEL_NUMBER_CHARACTERISTIC


async def _spin_until(predicate: Callable[[], bool]) -> None:
    """Yield to the event loop until predicate() holds (bounded)."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was never reached")


def _make_connected_lock_with_session(
    state_callback: Callable[[Iterable[LockStateValue]], None] = lambda _: None,
) -> Lock:
    """Build a connected Lock backed by a real Session over a mock BLE client.

    Mirrors tests/test_session.py: only cipher_encrypt is set, so notify frames
    pass through Session.decrypt unchanged and can be fed verbatim.
    """
    lock = _make_lock(state_callback)
    client = MagicMock()
    client.is_connected = True
    client.write_gatt_char = AsyncMock()
    lock.client = client
    lock.secure_session = MagicMock()
    session = Session(
        client, "mylock", asyncio.Lock(), set(), lock._internal_state_callback
    )
    session.cipher_encrypt = Cipher(
        algorithms.AES(bytes(16)),
        modes.CBC(bytes(16)),
    ).encryptor()
    lock.session = session
    return lock


# --------------------------------------------------------------------------- #
# Mechanical operations through the staged session wait (force_* + unlatch)
# --------------------------------------------------------------------------- #
def _with_checksum(frame: bytearray) -> bytes:
    """Stamp byte[3] so the 18-byte simple checksum sums to zero (a valid frame).

    Synthetic frames, not field captures: the operation-model replays in
    tests/test_lockoperation_model.py use verbatim captures, but the unlatch
    ack has no capture yet, so these are built to the layout the matchers key on.
    """
    frame = bytearray(frame)
    frame[0x03] = 0
    frame[0x03] = _simple_checksum(frame)
    return bytes(frame)


def _ack_frame(opcode: int, operation_byte: int) -> bytes:
    """A 0xAA acknowledgement echoing an operation's opcode and operation byte."""
    frame = bytearray(0x12)
    frame[0x00] = 0xAA
    frame[0x01] = opcode
    frame[0x04] = operation_byte
    return _with_checksum(frame)


def _op_response_frame(opcode: int, result: int = OperationError.COMM_SUCCESS) -> bytes:
    """A 0xBB op-response carrying the operation result in byte[15]."""
    frame = bytearray(0x12)
    frame[0x00] = 0xBB
    frame[0x01] = opcode
    frame[0x0F] = result
    return _with_checksum(frame)


async def _drive_operation(
    lock: Lock, op_attr: str, opcode: int, operation_byte: int
) -> bool:
    """Run a force_* method, feeding its ack then op-response through notify."""
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(0, bytearray(_ack_frame(opcode, operation_byte)))
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(opcode)))

    feeder = asyncio.create_task(feed())
    result: bool = await getattr(lock, op_attr)()
    await feeder
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op_attr", "opcode", "operation_byte"),
    [
        ("force_lock", Commands.LOCK, 0x00),
        ("force_unlock", Commands.UNLOCK, 0x00),
        ("force_securemode", Commands.LOCK, SECUREMODE_OPERATION_BYTE),
        ("force_unlatch", Commands.UNLOCK, UNLATCH_OPERATION_BYTE),
    ],
)
async def test_force_operations_complete_on_ack_then_op_response(
    op_attr: str, opcode: int, operation_byte: int
) -> None:
    """Each force_* completes only on its own ack, then its 0xBB op-response.

    Drives _execute_operation_command end to end through the real staged session
    wait; byte[15]=COMM_SUCCESS makes the method return True.
    """
    lock = _make_connected_lock_with_session()
    result = await _drive_operation(lock, op_attr, opcode, operation_byte)
    assert result is True


@pytest.mark.asyncio
async def test_public_unlatch_wrapper_runs_force_unlatch() -> None:
    """Lock.unlatch() always fires force_unlatch and completes on its op-response.

    Unlatch is momentary, so there is no steady "unlatched" state to
    short-circuit on the way lock()/unlock() do; the wrapper returns once the
    op-response arrives.
    """
    lock = _make_connected_lock_with_session()
    session = lock.session
    assert session is not None

    async def feed() -> None:
        await _spin_until(lambda: session._ack_future is not None)
        session._notify(
            0, bytearray(_ack_frame(Commands.UNLOCK, UNLATCH_OPERATION_BYTE))
        )
        await asyncio.sleep(0)
        session._notify(0, bytearray(_op_response_frame(Commands.UNLOCK)))

    feeder = asyncio.create_task(feed())
    # unlatch() returns None; the assertion is that it completes without raising.
    await lock.unlatch()
    await feeder


@pytest.mark.asyncio
async def test_force_unlatch_grants_the_extended_op_response_budget() -> None:
    """Unlatch gets the longer op-response budget because it is a longer motion.

    Unlatch drives the latch out (~2 s), dwells open (~6 s) and retracts (~2 s),
    so its op-response -- emitted when the motor stops -- lands ~10 s after the
    write, past the window a plain lock/unlock (~3 s) needs. force_unlatch passes
    UNLATCH_OPERATION_RESPONSE_TIMEOUT so the wait survives the full
    open-dwell-return motion, and encodes unlatch as the Unlock opcode with the
    unlatch operation byte (there is no dedicated unlatch opcode).
    """
    lock = _make_lock()
    session = MagicMock()

    def _build(opcode: int, cmd_byte: int) -> bytearray:
        cmd = bytearray(0x12)
        cmd[0x01] = opcode
        cmd[0x04] = cmd_byte
        return cmd

    session.build_operation_command.side_effect = _build
    lock.session = session
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    captured_command: bytearray | None = None
    captured_timeout: float | None = None

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        nonlocal captured_command, captured_timeout
        captured_command = command
        captured_timeout = response_timeout
        return True

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    assert await lock.force_unlatch() is True
    assert captured_timeout == UNLATCH_OPERATION_RESPONSE_TIMEOUT
    assert captured_command is not None
    assert captured_command[0x01] == Commands.UNLOCK
    assert captured_command[0x04] == UNLATCH_OPERATION_BYTE


@pytest.mark.asyncio
async def test_force_unlatch_failure_before_write_stays_retryable() -> None:
    """A failure before the command write leaves write_attempted False, so the
    error propagates unchanged and the caller may retry -- the latch never fired.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        raise DisconnectedError("dropped before the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(DisconnectedError):
        await lock.force_unlatch()


@pytest.mark.asyncio
async def test_force_unlatch_failure_after_write_converts_to_unlatch_error() -> None:
    """A retryable failure AFTER the command write converts to the non-retryable
    UnlatchError: a repeated unlatch fires the latch again, so it must not retry.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        assert progress is not None
        progress.write_attempted = True
        progress.command_written = True
        raise TimeoutError("no op-response after the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    # The originating error is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_force_unlatch_errored_write_converts_to_unlatch_error() -> None:
    """A write call that errors leaves delivery unknown (write_attempted True,
    command_written False): the request PDU may have reached the lock, so the
    failure converts to the non-retryable UnlatchError instead of retrying.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        assert progress is not None
        progress.write_attempted = True
        raise BleakError("link dropped during the write")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    # The originating error is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, BleakError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        EOFError("dbus socket closed"),
        BrokenPipeError("dbus socket closed"),
        AttributeError("backend went away mid-write"),
        ValueError("an error from nowhere near the retry set"),
    ],
)
async def test_force_unlatch_converts_every_post_write_failure(
    error: Exception,
) -> None:
    """The post-write conversion is type-blind, so no failure can re-send.

    Enumerating retryable types is what let this leak: the retry set is
    assembled from bleak_retry_connector and grows without reference to this
    guard. The first three of these were in that set and outside an earlier
    enumeration, so a post-write escape was retried and the latch fired
    twice; the fourth is in no retry set at all and converts just the same,
    which is the property that keeps the invariant true as the set widens.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        assert progress is not None
        progress.write_attempted = True
        raise error

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(UnlatchError) as excinfo:
        await lock.force_unlatch()
    assert excinfo.value.__cause__ is error


@pytest.mark.asyncio
async def test_force_unlatch_operation_incomplete_is_not_converted() -> None:
    """OperationIncompleteError is already non-retryable, so it propagates as
    itself even after the write -- it is not re-wrapped as UnlatchError.
    """
    lock = _make_lock()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    async def _fail(
        command: bytearray,
        command_name: str,
        response_timeout: float = OPERATION_RESPONSE_TIMEOUT,
        progress: OperationProgress | None = None,
    ) -> bool:
        assert progress is not None
        progress.command_written = True
        raise OperationIncompleteError("acked but no op-response")

    lock._execute_operation_command = _fail  # type: ignore[method-assign]
    with pytest.raises(OperationIncompleteError):
        await lock.force_unlatch()


@pytest.mark.parametrize("model", ["Yale Linus L2", "Yale Linus L2 Lite", "SL-103"])
def test_lock_info_can_open_true_for_open_support_models(model: str) -> None:
    """can_open is True for the Linus L2 family that supports the open action.

    "Yale Linus L2 Lite" is not a table entry, so it passes only with prefix
    matching; it pins the behaviour, not the table contents.
    """
    info = LockInfo("Yale/August", model, "serial", "1.0.0")
    assert info.can_open is True


@pytest.mark.parametrize("model", ["ASL-03", "YRD256", ""])
def test_lock_info_can_open_false_for_other_models(model: str) -> None:
    """can_open is False outside OPEN_SUPPORT_MODELS and for a blank model."""
    info = LockInfo("Yale/August", model, "serial", "1.0.0")
    assert info.can_open is False

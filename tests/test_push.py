import asyncio
import itertools
import logging
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakDBusError, BleakError

from yalexs_ble.const import (
    AuthState,
    AutoLockMode,
    AutoLockState,
    BatteryState,
    DoorStatus,
    LockInfo,
    LockState,
    LockStatus,
)
from yalexs_ble.lock import Lock
from yalexs_ble.push import (
    _AUTH_FAILURE_HISTORY,
    AUTH_FAILURE_TO_START_REAUTH,
    AUTO_LOCK_READ_FAILURE_BACKOFF,
    AUTO_LOCK_READ_FAILURE_THRESHOLD,
    AUTO_LOCK_READ_REFRESH_INTERVAL,
    AUTO_LOCK_READ_RESPONSE_TIMEOUT,
    AUTO_LOCK_WRITE_ATTEMPTS,
    BATTERY_REFRESH_INTERVAL,
    BATTERY_TIMEOUT_COOLDOWN,
    DEFAULT_ATTEMPTS,
    JAMMED_HOLD_TIME,
    NEVER_TIME,
    NO_BATTERY_SUPPORT_MODELS,
    POST_OP_RESPONSE_DEBOUNCE_DELAY,
    POST_OPERATION_BATTERY_COOLDOWN,
    SLOW_LATENCY,
    SLOW_MAX_INTERVAL,
    SLOW_MIN_INTERVAL,
    SLOW_TIMEOUT,
    YALE_MFR_ID,
    PushLock,
    operation_lock,
    retry_bluetooth_connection_error,
)
from yalexs_ble.session import (
    DisconnectedError,
    OperationIncompleteError,
    ResponseError,
    UnlatchError,
)

# Shared battery-supporting lock used across tests. model is NOT in
# NO_BATTERY_SUPPORT_MODELS, so the battery-workaround path is not taken.
TEST_LOCK_INFO = LockInfo(
    manufacturer="August",
    model="ASL-03",
    serial="12345",
    firmware="2.0.0",
)


@pytest.mark.asyncio
async def test_operation_lock():
    """Test the operation_lock function."""

    counter = 0

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        @operation_lock
        async def do_something(self):
            nonlocal counter
            counter += 1
            await asyncio.sleep(1)
            counter -= 1

    lock = MockPushLock()
    tasks = []
    for _ in range(10):
        tasks.append(asyncio.create_task(lock.do_something()))

    await asyncio.sleep(0)

    for _ in range(10):
        await asyncio.sleep(0)
        assert counter == 1

    for task in tasks:
        task.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_operation_lock_with_retry_bluetooth_connection_error():
    """Retry outside the operation lock: every attempt of every call runs
    under the lock, exactly one at a time, the lock is released between
    attempts, and the final error reaches the caller once the attempts are
    exhausted."""
    CALLS = 10
    real_sleep = asyncio.sleep
    active = 0
    max_active = 0
    calls: list[int] = []

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        async def _async_handle_disconnected(self, exc: Exception) -> None:
            """The retry wrapper awaits this hook on every retryable failure."""

        @retry_bluetooth_connection_error
        @operation_lock
        async def do_something(self, idx: int) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(idx)
            try:
                # Yield while inside the lock so an exclusion failure would
                # let a second call enter and be recorded in max_active. The
                # binding taken before the patch keeps this a real yield.
                await real_sleep(0)
                raise TimeoutError
            finally:
                active -= 1

    lock = MockPushLock()
    # Patch out the retry backoff sleep so the test stays event driven even
    # if the backoff policy widens to cover TimeoutError.
    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()):
        tasks = [asyncio.create_task(lock.do_something(idx)) for idx in range(CALLS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Every call ran its full attempt count and surfaced the final error.
    assert [type(result) for result in results] == [TimeoutError] * CALLS
    assert len(calls) == CALLS * DEFAULT_ATTEMPTS
    assert all(calls.count(idx) == DEFAULT_ATTEMPTS for idx in range(CALLS))
    # The lock serialized the attempts: no two calls were ever inside at once.
    assert max_active == 1
    assert active == 0
    # Retrying outside the lock releases it between attempts, so a call's
    # attempts are not contiguous: another call gets in before the retry.
    blocks = [
        calls[i : i + DEFAULT_ATTEMPTS] for i in range(0, len(calls), DEFAULT_ATTEMPTS)
    ]
    assert not any(len(set(block)) == 1 for block in blocks)


@pytest.mark.asyncio
async def test_retry_bluetooth_connection_error_with_operation_lock():
    """The operation lock outside the retry wrapper: a call holds the lock
    across its whole retry loop, so its attempts run back to back before the
    next call starts, and the final error reaches the caller."""
    CALLS = 10
    real_sleep = asyncio.sleep
    active = 0
    max_active = 0
    calls: list[int] = []

    class MockPushLock:
        def __init__(self):
            self._operation_lock = asyncio.Lock()

        @property
        def name(self):
            return "lock"

        async def _async_handle_disconnected(self, exc: Exception) -> None:
            """The retry wrapper awaits this hook on every retryable failure."""

        @operation_lock
        @retry_bluetooth_connection_error
        async def do_something(self, idx: int) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(idx)
            try:
                # Yield while inside the lock so an exclusion failure would
                # let a second call enter and be recorded in max_active. The
                # binding taken before the patch keeps this a real yield.
                await real_sleep(0)
                raise TimeoutError
            finally:
                active -= 1

    lock = MockPushLock()
    # Patch out the retry backoff sleep so the test stays event driven even
    # if the backoff policy widens to cover TimeoutError.
    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()):
        tasks = [asyncio.create_task(lock.do_something(idx)) for idx in range(CALLS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Every call ran its full attempt count and surfaced the final error.
    assert [type(result) for result in results] == [TimeoutError] * CALLS
    assert len(calls) == CALLS * DEFAULT_ATTEMPTS
    assert all(calls.count(idx) == DEFAULT_ATTEMPTS for idx in range(CALLS))
    # The lock serialized the attempts: no two calls were ever inside at once.
    assert max_active == 1
    assert active == 0
    # Holding the lock across the retry loop keeps a call's attempts
    # contiguous: each consecutive block of DEFAULT_ATTEMPTS entries in the
    # call order belongs to a single call.
    blocks = [
        calls[i : i + DEFAULT_ATTEMPTS] for i in range(0, len(calls), DEFAULT_ATTEMPTS)
    ]
    assert all(len(set(block)) == 1 for block in blocks)


def test_needs_battery_workaround():
    assert "SL-103" in NO_BATTERY_SUPPORT_MODELS
    assert "CERES" in NO_BATTERY_SUPPORT_MODELS
    assert "Yale Linus L2" in NO_BATTERY_SUPPORT_MODELS
    assert "ASL-03" not in NO_BATTERY_SUPPORT_MODELS
    assert "MD-04I" not in NO_BATTERY_SUPPORT_MODELS


@pytest.mark.asyncio
async def test_update_continues_after_battery_timeout():
    """
    Test that _update() continues and completes successfully
    even when battery() times out.

    Requirements:
    - battery() timeout does not fail entire update
    - lock_status/door_status/auto_lock_status still get called
    - final state has valid lock/door values (not UNKNOWN)
    - no forced disconnect due to battery timeout
    """

    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    # Mock lock that times out on battery()
    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(return_value=TEST_LOCK_INFO)

    # Battery times out
    mock_lock.battery = AsyncMock(side_effect=TimeoutError("Battery timeout"))

    # But other calls succeed
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True

    # Mock advertisement_data for connection_info
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        # Should NOT raise exception
        final_state = await push_lock._update()

        # Battery call was attempted
        mock_lock.battery.assert_called_once()

        # Other status calls still happened
        mock_lock.door_status.assert_called_once()
        mock_lock.auto_lock_status.assert_called_once()
        mock_lock.lock_status.assert_called_once()

        # Final state has valid lock/door (from the successful calls)
        assert final_state.lock == LockStatus.LOCKED
        assert final_state.door == DoorStatus.CLOSED

        # Battery should be None since it timed out
        assert final_state.battery is None


@pytest.mark.asyncio
async def test_poll_battery_cooldown_skip():
    """Test that _poll_battery skips when on cooldown."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    # Set cooldown to 5 seconds in the future
    push_lock._earliest_battery_attempt_time = time.monotonic() + 5.0

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should skip the request
    assert made_request is False
    mock_lock.battery.assert_not_called()
    # State should be unchanged
    assert result_state == initial_state


@pytest.mark.asyncio
async def test_poll_battery_success():
    """Test that _poll_battery successfully fetches battery and resets cooldown."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    # Set cooldown to simulate previous timeout
    push_lock._earliest_battery_attempt_time = time.monotonic() + 100.0

    mock_lock = MagicMock()
    battery_state = BatteryState(voltage=6.0, percentage=80)
    mock_lock.battery = AsyncMock(return_value=battery_state)

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery (cooldown should be ignored since it's in the future)
    # Wait a moment to ensure cooldown expires
    push_lock._earliest_battery_attempt_time = NEVER_TIME

    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should have battery data
    assert result_state.battery == battery_state
    assert result_state.auth is not None
    assert result_state.auth.successful is True

    # Cooldown should be reset to NEVER_TIME
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_poll_battery_bleak_error():
    """Test that _poll_battery handles BleakError gracefully."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(side_effect=BleakError("Connection failed"))

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should be unchanged (error was logged but not raised)
    assert result_state == initial_state

    # Cooldown should NOT be set (only TimeoutError sets cooldown)
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_poll_battery_bleak_dbus_error():
    """Test that _poll_battery handles BleakDBusError gracefully."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(
        side_effect=BleakDBusError("DBus error", "error body")
    )

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Call _poll_battery
    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Should make the request
    assert made_request is True
    mock_lock.battery.assert_called_once()

    # State should be unchanged (error was logged but not raised)
    assert result_state == initial_state

    # Cooldown should NOT be set (only TimeoutError sets cooldown)
    assert push_lock._earliest_battery_attempt_time == NEVER_TIME


@pytest.mark.asyncio
async def test_update_preserves_notify_state_from_cache() -> None:
    """
    Test that _update() does not overwrite lock/door state with UNKNOWN
    when notify callbacks have updated the cached state.

    Regression test for race condition where:
    1. Update starts with UNKNOWN state
    2. Notify callback updates cached state to LOCKED/CLOSED during update
    3. Update skips polling lock_status (already seen this session)
    4. Final state should preserve LOCKED/CLOSED from cache, not revert to UNKNOWN
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    # Start with UNKNOWN state; update will normally leave it UNKNOWN
    push_lock._lock_state = LockState(
        lock=LockStatus.UNKNOWN,
        door=DoorStatus.UNKNOWN,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Mock lock that doesn't return lock/door (simulating skipped polling)
    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(
        return_value=MagicMock(model="ASL-03", door_sense=True)
    )

    push_lock._lock_info = MagicMock(model="ASL-03", door_sense=True)
    push_lock._running = True

    # Mark lock/door/battery as already seen to simulate skipped polling
    push_lock._seen_this_session.add(LockStatus)
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)

    # Mock advertisement_data for connection_info
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    # Gate auto_lock_status so we can inject notify updates mid-_update
    auto_lock_in_progress = asyncio.Event()
    allow_auto_lock_to_continue = asyncio.Event()

    async def auto_lock_status():
        auto_lock_in_progress.set()
        await allow_auto_lock_to_continue.wait()
        return AutoLockState(mode=AutoLockMode.OFF, duration=0)

    mock_lock.auto_lock_status = AsyncMock(side_effect=auto_lock_status)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        update_task = asyncio.create_task(push_lock._update())

        # Wait until _update is awaiting auto_lock_status, then simulate notify callback
        await auto_lock_in_progress.wait()
        push_lock._update_any_state([LockStatus.LOCKED, DoorStatus.CLOSED])
        allow_auto_lock_to_continue.set()

        final_state = await update_task

        # The critical assertion: lock/door must be preserved from cache
        assert final_state.lock == LockStatus.LOCKED, (
            f"Lock status should be LOCKED from cache, got {final_state.lock}"
        )
        assert final_state.door == DoorStatus.CLOSED, (
            f"Door status should be CLOSED from cache, got {final_state.door}"
        )


@pytest.mark.asyncio
async def test_update_auto_lock_from_notify_path_survives_poll_result() -> None:
    """_update() carries the notify-published auto-lock into its final state.

    The auto-lock read's return value is the READSETTING acknowledgment
    constant (OFF) and is discarded; the stored setting arrives as the 0xBB
    settings response on the notify path during the cycle. The end-of-update
    restore must apply the notify-published value, not revert to the cycle's
    starting snapshot or the poll constant.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    push_lock._lock_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    mock_lock = MagicMock()
    push_lock._lock_info = MagicMock(model="ASL-03", door_sense=True)
    push_lock._running = True

    # Mark everything but AutoLockState seen so only the auto-lock read runs.
    push_lock._seen_this_session.add(LockStatus)
    push_lock._seen_this_session.add(DoorStatus)
    push_lock._seen_this_session.add(BatteryState)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    # Gate the read so the settings-response notify publish lands mid-cycle.
    auto_lock_in_progress = asyncio.Event()
    allow_auto_lock_to_continue = asyncio.Event()

    async def auto_lock_status():
        auto_lock_in_progress.set()
        await allow_auto_lock_to_continue.wait()
        # The acknowledgment constant -- must not reach the final state.
        return AutoLockState(mode=AutoLockMode.OFF, duration=0)

    mock_lock.auto_lock_status = AsyncMock(side_effect=auto_lock_status)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        update_task = asyncio.create_task(push_lock._update())

        await auto_lock_in_progress.wait()
        # The 0xBB settings response publishing through the notify path.
        push_lock._update_any_state([AutoLockState(AutoLockMode.TIMER, 1800)])
        allow_auto_lock_to_continue.set()

        final_state = await update_task

        assert final_state.auto_lock == AutoLockState(AutoLockMode.TIMER, 1800), (
            f"Auto-lock should be the notify-published value, "
            f"got {final_state.auto_lock}"
        )


@pytest.mark.asyncio
async def test_update_continues_when_lock_info_probe_fails() -> None:
    """Test that _update() proceeds with defaults when lock_info() raises."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(side_effect=TimeoutError("probe timed out"))
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        final_state = await push_lock._update()

    # lock_info was attempted
    mock_lock.lock_info.assert_called_once()

    # Update still completed with real data
    assert final_state.lock == LockStatus.LOCKED

    # door_status not called because model="" makes door_sense=False
    mock_lock.door_status.assert_not_called()
    assert final_state.door == DoorStatus.UNKNOWN

    # Defaults were used for lock_info, serial falls back to MAC address
    assert push_lock._lock_info is not None
    assert push_lock._lock_info.model == ""
    assert push_lock._lock_info.serial == "aa:bb:cc:dd:ee:ff"
    assert push_lock._lock_info.door_sense is False


@pytest.mark.asyncio
async def test_update_continues_when_lock_info_probe_bleak_error() -> None:
    """Test that _update() proceeds with defaults when lock_info() raises BleakError."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(
        side_effect=BleakError("connection dropped during probe")
    )
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        final_state = await push_lock._update()

    assert final_state.lock == LockStatus.LOCKED
    assert push_lock._lock_info is not None
    assert push_lock._lock_info.manufacturer == "Unknown"
    assert push_lock._lock_info.serial == "aa:bb:cc:dd:ee:ff"
    assert push_lock._lock_info.door_sense is False


@pytest.mark.asyncio
async def test_update_sets_slow_connection_params_when_always_connected():
    """Test _update() sets slow BLE connection params when always connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock()

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    mock_client.set_connection_params.assert_called_once_with(
        SLOW_MIN_INTERVAL, SLOW_MAX_INTERVAL, SLOW_LATENCY, SLOW_TIMEOUT
    )


def test_slow_connection_params_are_latency_based():
    """Slow mode must idle via peripheral latency, not a long interval.

    A long min == max interval throttles notification delivery (one frame per
    two connection events), so an operation's reply cannot drain before the
    next command is written. The idle duty cycle has to come from latency
    while the base interval stays short.
    """
    # Short base interval so notifications drain quickly once latency drops.
    assert SLOW_MAX_INTERVAL * 1.25 <= 50  # ms
    assert SLOW_MIN_INTERVAL == SLOW_MAX_INTERVAL
    # Idle wake-up is ~500ms, which is where the battery saving comes from.
    assert 400 <= (1 + SLOW_LATENCY) * SLOW_MAX_INTERVAL * 1.25 <= 600  # ms
    # Core spec: supervision timeout > (1 + latency) * max_interval * 2.
    assert SLOW_TIMEOUT * 10 > (1 + SLOW_LATENCY) * SLOW_MAX_INTERVAL * 1.25 * 2


@pytest.mark.asyncio
async def test_update_does_not_set_connection_params_when_not_always_connected():
    """Test _update() skips connection params when not always connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock()

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    mock_client.set_connection_params.assert_not_called()


@pytest.mark.asyncio
async def test_update_handles_connection_params_failure():
    """Test that _update() continues even if set_connection_params fails."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_client = MagicMock()
    mock_client.set_connection_params = AsyncMock(
        side_effect=BleakError("Failed to set params")
    )

    mock_lock = MagicMock()
    mock_lock.client = mock_client
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=5.5, percentage=95))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.auto_lock_status = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        # Should NOT raise even though set_connection_params failed
        final_state = await push_lock._update()

    assert final_state.lock == LockStatus.LOCKED
    mock_client.set_connection_params.assert_called_once()


@pytest.mark.asyncio
async def test_battery_refresh_clears_seen_and_repoll_when_due():
    """In always_connected mode, _update() should evict BatteryState from
    _seen_this_session and re-poll battery once BATTERY_REFRESH_INTERVAL
    has elapsed since the last refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    battery_state = BatteryState(voltage=4.0, percentage=90)
    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(return_value=battery_state)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already polled this session
    push_lock._seen_this_session.add(BatteryState)

    # Set the refresh deadline in the past so a refresh is due
    push_lock._next_battery_refresh_time = time.monotonic() - 1.0
    before_update = time.monotonic()

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        final_state = await push_lock._update()

    # Battery should have been re-polled
    mock_lock.battery.assert_called_once()
    assert final_state.battery == battery_state
    # Deadline should have been pushed out a full interval from the poll
    assert (
        push_lock._next_battery_refresh_time >= before_update + BATTERY_REFRESH_INTERVAL
    )


@pytest.mark.asyncio
async def test_battery_refresh_not_due_skips_repoll():
    """In always_connected mode, _update() should NOT re-poll battery when
    BATTERY_REFRESH_INTERVAL has not yet elapsed."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already polled this session
    push_lock._seen_this_session.add(BatteryState)

    # Set the refresh deadline in the future — not yet due
    push_lock._next_battery_refresh_time = time.monotonic() + BATTERY_REFRESH_INTERVAL

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # Battery should NOT have been re-polled
    mock_lock.battery.assert_not_called()


@pytest.mark.asyncio
async def test_battery_refresh_does_not_fire_when_not_always_connected():
    """The periodic battery refresh must not affect non-always-connected locks.
    In normal mode _seen_this_session clears on each new connection, so battery
    is polled naturally and the interval guard must stay dormant."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock()
    mock_lock.client = MagicMock()
    mock_lock.client.set_connection_params = AsyncMock()

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    push_lock._running = True

    # Simulate battery already seen and a refresh deadline in the past
    push_lock._seen_this_session.add(BatteryState)
    refresh_deadline = time.monotonic() - 1.0
    push_lock._next_battery_refresh_time = refresh_deadline

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # Refresh block should not have fired — battery skipped because it is
    # in _seen_this_session and always_connected is False
    mock_lock.battery.assert_not_called()
    # Deadline must not have been touched
    assert push_lock._next_battery_refresh_time == refresh_deadline


@pytest.mark.asyncio
async def test_battery_refresh_due_but_on_cooldown_does_not_evict():
    """A refresh that comes due while the battery cooldown is active must not
    evict BatteryState or poll early. The cooldown gate precedes eviction, so
    BatteryState stays in _seen_this_session and the deadline is untouched until
    a later cycle (after cooldown) can actually re-poll — never an early poll."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:ff",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock()

    initial_state = LockState(
        lock=LockStatus.LOCKED,
        door=DoorStatus.CLOSED,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )

    # Battery already polled this session and the refresh is due...
    push_lock._seen_this_session.add(BatteryState)
    refresh_deadline = time.monotonic() - 1.0
    push_lock._next_battery_refresh_time = refresh_deadline
    # ...but a prior timeout left the battery cooldown active.
    push_lock._earliest_battery_attempt_time = time.monotonic() + 100.0

    result_state, made_request = await push_lock._poll_battery(mock_lock, initial_state)

    # Cooldown gate wins: no poll, no eviction, deadline untouched.
    assert made_request is False
    mock_lock.battery.assert_not_called()
    assert BatteryState in push_lock._seen_this_session
    assert push_lock._next_battery_refresh_time == refresh_deadline
    assert result_state == initial_state


@pytest.mark.asyncio
async def test_disconnected_callback_schedules_reconnect_when_always_connected() -> (
    None
):
    """Disconnect callback schedules keep-alive when always_connected and auth ok."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:01",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    _AUTH_FAILURE_HISTORY.auth_success(push_lock.address)

    with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
        push_lock._disconnected_callback()

    mock_keep_alive.assert_called_once()


@pytest.mark.asyncio
async def test_disconnected_callback_skips_reconnect_after_auth_failures() -> None:
    """Disconnect callback skips keep-alive when auth has failed enough times."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:02",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    for _ in range(AUTH_FAILURE_TO_START_REAUTH):
        _AUTH_FAILURE_HISTORY.auth_failed(push_lock.address)

    try:
        with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
            push_lock._disconnected_callback()
        mock_keep_alive.assert_not_called()
    finally:
        _AUTH_FAILURE_HISTORY.auth_success(push_lock.address)


@pytest.mark.asyncio
async def test_disconnected_callback_noop_when_not_always_connected() -> None:
    """Disconnect callback does nothing in non-always-connected mode."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:03",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with patch.object(push_lock, "_keep_alive") as mock_keep_alive:
        push_lock._disconnected_callback()

    mock_keep_alive.assert_not_called()


@pytest.mark.asyncio
async def test_keep_alive_noop_when_not_always_connected() -> None:
    """Keep-alive returns immediately when not always_connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:04",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_schedule_future_update") as mock_schedule_update,
        patch.object(push_lock, "_schedule_next_keep_alive") as mock_next_keep_alive,
    ):
        push_lock._keep_alive()

    mock_schedule_update.assert_not_called()
    mock_next_keep_alive.assert_not_called()


@pytest.mark.asyncio
async def test_keep_alive_schedules_update_and_next_when_always_connected() -> None:
    """Keep-alive schedules update and next keep-alive when always_connected."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:05",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_schedule_future_update") as mock_schedule_update,
        patch.object(push_lock, "_schedule_next_keep_alive") as mock_next_keep_alive,
    ):
        push_lock._keep_alive()

    mock_schedule_update.assert_called_once_with(0)
    mock_next_keep_alive.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_with_timer_skips_when_operation_lock_held() -> None:
    """Disconnect timer reschedules itself when an operation is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:06",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async with push_lock._operation_lock:
        with (
            patch.object(push_lock, "_reset_disconnect_timer") as mock_reset,
            patch.object(push_lock, "background_task") as mock_bg,
        ):
            push_lock._disconnect_with_timer(5.0)

    mock_reset.assert_called_once()
    mock_bg.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_with_timer_runs_deferred_update_when_pending() -> None:
    """Disconnect timer cancels future update and runs it when one is pending."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:07",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    # Simulate a pending deferred update without actually scheduling on the loop
    push_lock._cancel_deferred_update = MagicMock()

    with (
        patch.object(push_lock, "_reset_disconnect_timer") as mock_reset,
        patch.object(push_lock, "_cancel_future_update") as mock_cancel_future,
        patch.object(push_lock, "_deferred_update") as mock_deferred,
        patch.object(push_lock, "background_task") as mock_bg,
    ):
        push_lock._disconnect_with_timer(5.0)

    mock_reset.assert_called_once()
    mock_cancel_future.assert_called_once()
    mock_deferred.assert_called_once()
    mock_bg.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_with_timer_triggers_disconnect_when_idle() -> None:
    """Disconnect timer schedules a forced disconnect when idle."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:08",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
        patch.object(push_lock, "background_task") as mock_bg,
    ):
        push_lock._disconnect_with_timer(5.0)
        # Close the coroutine that would have been scheduled, to avoid
        # an unawaited-coroutine warning at GC time.
        (coro,), _ = mock_bg.call_args
        coro.close()

    mock_cancel.assert_called_once()
    mock_bg.assert_called_once()


@pytest.mark.asyncio
async def test_async_handle_disconnected_skips_when_connect_in_progress() -> None:
    """Handle-disconnected returns early when a connect is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:09",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    async with push_lock._connect_lock:
        with (
            patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
            patch.object(
                push_lock, "_execute_disconnect", new_callable=AsyncMock
            ) as mock_disconnect,
        ):
            await push_lock._async_handle_disconnected(RuntimeError("boom"))

    mock_cancel.assert_not_called()
    mock_disconnect.assert_not_called()


@pytest.mark.asyncio
async def test_async_handle_disconnected_executes_disconnect_when_idle() -> None:
    """Handle-disconnected runs full cleanup when no connect is in progress."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0a",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with (
        patch.object(push_lock, "_cancel_disconnect_timer") as mock_cancel,
        patch.object(
            push_lock, "_execute_disconnect", new_callable=AsyncMock
        ) as mock_disconnect,
    ):
        await push_lock._async_handle_disconnected(RuntimeError("boom"))

    mock_cancel.assert_called_once()
    mock_disconnect.assert_called_once()


class _MockRetryLock:
    """Minimal PushLock surface needed by retry_bluetooth_connection_error."""

    def __init__(self) -> None:
        self.address = "aa:bb:cc:dd:ee:ff"
        self._async_handle_disconnected = AsyncMock()

    @property
    def name(self) -> str:
        return "lock"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        BleakDBusError("org.bluez.Error.Failed", []),
        DisconnectedError("disconnected"),
        BleakError("bleak error"),
        TimeoutError(),
        ResponseError("response"),
    ],
)
async def test_retry_eventually_succeeds_for_all_retryable_exceptions(
    exc: Exception,
) -> None:
    """All retryable exceptions get retried, then succeed on a later attempt."""
    lock = _MockRetryLock()
    calls = 0

    @retry_bluetooth_connection_error
    async def op(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise exc
        return "ok"

    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()):
        result = await op(lock)

    assert result == "ok"
    assert calls == 2
    lock._async_handle_disconnected.assert_awaited_once_with(exc)


@pytest.mark.asyncio
async def test_retry_disconnected_error_reraised_unchanged_at_max_attempts() -> None:
    """DisconnectedError is not a BleakError, so it re-raises unchanged."""
    lock = _MockRetryLock()
    err = DisconnectedError("gone")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(DisconnectedError) as exc_info,
    ):
        await op(lock)

    assert exc_info.value is err
    # Called once per attempt.
    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_disconnect_bleak_error_converted_to_disconnected_error() -> None:
    """A BleakError that reads as a disconnect converts to DisconnectedError."""
    lock = _MockRetryLock()
    err = BleakError("device disconnected")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(DisconnectedError) as exc_info,
    ):
        await op(lock)

    assert exc_info.value.__cause__ is err
    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_bleak_error_raises_at_max_attempts() -> None:
    """Non-disconnect retryable exceptions propagate their original type."""
    lock = _MockRetryLock()
    err = BleakError("nope")

    @retry_bluetooth_connection_error
    async def op(self):
        raise err

    with (
        patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()),
        pytest.raises(BleakError),
    ):
        await op(lock)

    assert lock._async_handle_disconnected.await_count == DEFAULT_ATTEMPTS


@pytest.mark.asyncio
async def test_retry_backoff_exceptions_sleep_between_attempts() -> None:
    """RETRY_BACKOFF_EXCEPTIONS pause 0.25s between retries; others do not."""
    lock = _MockRetryLock()

    @retry_bluetooth_connection_error
    async def op_backoff(self):
        raise BleakDBusError("org.bluez.Error.Failed", [])

    @retry_bluetooth_connection_error
    async def op_nobackoff(self):
        raise TimeoutError

    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(BleakError):
            await op_backoff(lock)
        backoff_calls = list(sleep_mock.await_args_list)

    # Sleeps happen only between non-final attempts.
    assert backoff_calls == [call(0.25)] * (DEFAULT_ATTEMPTS - 1)

    lock2 = _MockRetryLock()
    with patch("yalexs_ble.push.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(TimeoutError):
            await op_nobackoff(lock2)
        assert sleep_mock.await_args_list == []


@pytest.mark.asyncio
@pytest.mark.parametrize("setter", ["set_auto_lock_duration", "set_auto_lock_mode"])
async def test_set_auto_lock_timeout_warns_and_names_the_failure(
    caplog: pytest.LogCaptureFixture, setter: str
) -> None:
    """An unconfirmed auto lock write warns once and re-raises with a message."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0b",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    with patch.object(push_lock, "_set_auto_lock", new_callable=AsyncMock) as mock_set:
        mock_set.side_effect = TimeoutError()
        with pytest.raises(TimeoutError) as exc_info:
            if setter == "set_auto_lock_duration":
                await push_lock.set_auto_lock_duration(30)
            else:
                await push_lock.set_auto_lock_mode(AutoLockMode.TIMER)

    assert "Lock did not confirm the auto lock setting write" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "the lock may not support auto lock" in warnings[0].getMessage()
    assert f"after {AUTO_LOCK_WRITE_ATTEMPTS} attempts" in warnings[0].getMessage()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setter", "arg", "auto_lock", "auto_lock_prev", "expected"),
    [
        # Turning auto lock off writes OFF with a zero duration.
        ("set_auto_lock_mode", AutoLockMode.OFF, None, None, (AutoLockMode.OFF, 0)),
        ("set_auto_lock_duration", 0, None, None, (AutoLockMode.OFF, 0)),
        # Already off returns without writing.
        (
            "set_auto_lock_mode",
            AutoLockMode.OFF,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            None,
            None,
        ),
        (
            "set_auto_lock_duration",
            0,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            None,
            None,
        ),
        # A mode change keeps the current duration; a duration change keeps
        # the current mode.
        (
            "set_auto_lock_mode",
            AutoLockMode.INSTANT,
            AutoLockState(mode=AutoLockMode.TIMER, duration=120),
            None,
            (AutoLockMode.INSTANT, 120),
        ),
        (
            "set_auto_lock_duration",
            60,
            AutoLockState(mode=AutoLockMode.INSTANT, duration=5),
            None,
            (AutoLockMode.INSTANT, 60),
        ),
        # When auto lock is currently off, fall back to the previous state.
        (
            "set_auto_lock_mode",
            AutoLockMode.TIMER,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            AutoLockState(mode=AutoLockMode.TIMER, duration=300),
            (AutoLockMode.TIMER, 300),
        ),
        (
            "set_auto_lock_duration",
            60,
            AutoLockState(mode=AutoLockMode.OFF, duration=0),
            AutoLockState(mode=AutoLockMode.INSTANT, duration=10),
            (AutoLockMode.INSTANT, 60),
        ),
    ],
)
async def test_set_auto_lock_wrappers_choose_the_written_pair(
    setter: str,
    arg: AutoLockMode | int,
    auto_lock: AutoLockState | None,
    auto_lock_prev: AutoLockState | None,
    expected: tuple[AutoLockMode, int] | None,
) -> None:
    """The public setters pick mode and duration from current, then previous state."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0c",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    if auto_lock or auto_lock_prev:
        push_lock._lock_state = LockState(
            lock=LockStatus.LOCKED,
            door=DoorStatus.CLOSED,
            battery=None,
            auth=None,
            auto_lock=auto_lock,
            auto_lock_prev=auto_lock_prev,
        )

    with patch.object(push_lock, "_set_auto_lock", new_callable=AsyncMock) as mock_set:
        await getattr(push_lock, setter)(arg)

    if expected is None:
        mock_set.assert_not_awaited()
    else:
        mock_set.assert_awaited_once_with(*expected)


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_backoff_arms_after_threshold_timeouts(
    always_connected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """The read backs off only after THRESHOLD consecutive timeouts, not before.

    The arm path is mode-independent, so it is exercised with and without
    always_connected for symmetry with the response-timeout arming test.
    """
    caplog.set_level(logging.INFO)
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=always_connected,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    # Below the threshold: each timeout is counted, but the backoff is not armed.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        assert await push_lock._read_auto_lock_setting(mock_lock) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert not [
        r for r in caplog.records if "may not support auto lock" in r.getMessage()
    ]

    # The threshold-th consecutive timeout arms the backoff and logs once.
    # Arming restarts the count, so the field reads zero afterwards.
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    latch = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO and "may not support auto lock" in r.getMessage()
    ]
    assert len(latch) == 1
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD

    # Now backed off: the read is skipped without ever touching the lock.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_auto_lock_read_backoff_reearned_after_window() -> None:
    """When the backoff window expires the count restarts and is re-earned."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    # Arriving as if a prior window has just armed and reset: no failures held,
    # and the window is already past so reads resume.
    push_lock._auto_lock_read_ack_failures = 0
    push_lock._earliest_auto_lock_read_time = NEVER_TIME

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    # A fresh run of consecutive timeouts is needed to arm the backoff again.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        assert await push_lock._read_auto_lock_setting(mock_lock) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    assert mock_lock.auto_lock_status.await_count == AUTO_LOCK_READ_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_auto_lock_read_success_resets_failure_count() -> None:
    """A settings response arriving clears the failures and arms the refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0d",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD - 1
    push_lock._auto_lock_read_response_failures = 1
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = time.monotonic() + 10.0
    push_lock._earliest_auto_lock_read_time = time.monotonic() + 100.0

    before = time.monotonic()
    push_lock._update_any_state([AutoLockState(mode=AutoLockMode.TIMER, duration=30)])

    # The value landing -- not the read call returning -- is the success signal:
    # it clears both failure counts and disarms the pending-response deadline.
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert (
        push_lock._next_auto_lock_read_time >= before + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert AutoLockState in push_lock._seen_this_session
    assert push_lock.auto_lock == AutoLockState(mode=AutoLockMode.TIMER, duration=30)


@pytest.mark.asyncio
async def test_auto_lock_read_backoff_survives_reconnect() -> None:
    """The failure backoff outlives a reconnect -- the W2 regression guard."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0e",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    deadline = time.monotonic() + AUTO_LOCK_READ_FAILURE_BACKOFF
    push_lock._earliest_auto_lock_read_time = deadline
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.connect = AsyncMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        client = await push_lock._ensure_connected()

    # The reconnect cleared _seen_this_session but must NOT clear the backoff.
    assert AutoLockState not in push_lock._seen_this_session
    assert push_lock._earliest_auto_lock_read_time == deadline
    assert push_lock._auto_lock_read_ack_failures == AUTO_LOCK_READ_FAILURE_THRESHOLD

    # Still backed off after the reconnect: the read stays skipped, so the
    # case-2 storm cannot restart.
    assert await push_lock._read_auto_lock_setting(client) is False
    mock_lock.auto_lock_status.assert_not_called()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_refresh_evicts_only_when_due() -> None:
    """The refresh re-reads only after the interval, mirroring the battery refresh."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:0f",
        key="0800200c9a66",
        key_index=1,
        always_connected=True,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    push_lock._seen_this_session.add(AutoLockState)

    # Not yet due: the seen gate holds, no re-read.
    push_lock._next_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert AutoLockState in push_lock._seen_this_session
    mock_lock.auto_lock_status.assert_not_called()

    # Deadline passed: evict AutoLockState and issue a fresh read.
    push_lock._next_auto_lock_read_time = time.monotonic() - 1.0
    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    mock_lock.auto_lock_status.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_lock_read_no_evict_outside_always_connected() -> None:
    """Outside always_connected mode the refresh never evicts a seen value."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:10",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    push_lock._seen_this_session.add(AutoLockState)
    # Deadline long past, but the eviction is gated on always_connected.
    push_lock._next_auto_lock_read_time = time.monotonic() - 1.0

    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert AutoLockState in push_lock._seen_this_session
    mock_lock.auto_lock_status.assert_not_called()


@pytest.mark.asyncio
async def test_auto_lock_read_transport_error_does_not_arm_backoff() -> None:
    """A transport fault mirrors _poll_battery: skip without arming the backoff."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:14",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=BleakError("boom"))

    # A non-timeout fault is not the "alive but silent" signature: the read is
    # skipped, but the failure count and backoff are left untouched.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME


@pytest.mark.asyncio
async def test_auto_lock_read_timeout_does_not_propagate_out_of_update() -> None:
    """A read timeout is caught in the helper, so _update completes normally."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:11",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(return_value=TEST_LOCK_INFO)
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        state = await push_lock._update()

    # The update returned instead of raising: no forced disconnect, and the
    # timeout was counted as a failure rather than propagated.
    assert state.lock == LockStatus.LOCKED
    assert push_lock._auto_lock_read_ack_failures == 1
    mock_lock.auto_lock_status.assert_awaited_once()
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_set_auto_lock_write_resets_read_backoff() -> None:
    """A confirmed write clears the backoff and evicts the seen value."""
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:12",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True
    push_lock._auto_lock_read_ack_failures = AUTO_LOCK_READ_FAILURE_THRESHOLD
    push_lock._auto_lock_read_response_failures = 2
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = time.monotonic() + 10.0
    push_lock._earliest_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._next_auto_lock_read_time = (
        time.monotonic() + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.set_auto_lock = AsyncMock()

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        patch.object(push_lock, "_complete_operation"),
    ):
        await push_lock._set_auto_lock(AutoLockMode.TIMER, 30)

    mock_lock.set_auto_lock.assert_awaited_once_with(AutoLockMode.TIMER, 30)
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    assert push_lock._earliest_auto_lock_read_time == NEVER_TIME
    assert push_lock._next_auto_lock_read_time == NEVER_TIME
    assert AutoLockState not in push_lock._seen_this_session

    # With the value evicted and no backoff, the next read is issued again.
    mock_lock.auto_lock_status = AsyncMock(return_value=None)
    assert await push_lock._read_auto_lock_setting(mock_lock) is True


# ---------------------------------------------------------------------------
# Auto lock read: the four settings-command outcomes (see
# notes/yale/autolock_settings_command_outcome_taxonomy.md), each with and
# without always_connected, plus dropout and advert-driven connect-on-demand.
#
#   Case 1 -- dead lock, answers nothing. Caught by the earlier unguarded reads
#            in _update, so the auto lock read is never reached.
#   Case 2 -- alive but silent to the read: the ack times out.
#   Case 3 -- acks the read but withholds the 0xBB value: the response window
#            lapses with the value unseen.
#   Case 4 -- full working lock: ack, then the 0xBB value on the notify path.
# ---------------------------------------------------------------------------


def _auto_lock_push_lock(address: str, *, always_connected: bool) -> PushLock:
    """A named PushLock for the auto lock read outcome tests."""
    push_lock = PushLock(
        address=address,
        key="0800200c9a66",
        key_index=1,
        always_connected=always_connected,
    )
    push_lock._name = "Test Lock"
    return push_lock


def _auto_lock_update_lock(auto_lock_status: AsyncMock) -> MagicMock:
    """A mock Lock answering every read so _update reaches the auto lock read.

    The auto lock read itself is wired per the outcome under test.
    """
    lock = MagicMock()
    lock.connect = AsyncMock()
    lock.is_connected = True
    lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))
    lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    lock.auto_lock_status = auto_lock_status
    return lock


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_response_timeout_arms_backoff(
    always_connected: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Case 3: the lock acks the read but withholds the 0xBB value.

    The read completes on the ack, so no timeout fires; the pending-response
    deadline lapses on the next cycle with the value still unseen. After
    THRESHOLD such response timeouts in a row the read backs off, and the INFO
    log reports the ack and response counts separately.
    """
    caplog.set_level(logging.INFO)
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:20", always_connected=always_connected
    )
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)  # ack ok, no 0xBB

    # First cycle issues the read and arms the pending-response deadline.
    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_read_response_failures == 0

    # Each later cycle finds the window lapsed with the value still unseen: a
    # response timeout, which re-reads until the threshold is reached.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
        assert await push_lock._read_auto_lock_setting(mock_lock) is True
        assert push_lock._auto_lock_read_response_failures == expected
        assert push_lock._auto_lock_read_ack_failures == 0
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    # The threshold-th response timeout arms the backoff and logs the breakdown.
    push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_response_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    latch = [
        r
        for r in caplog.records
        if r.levelno == logging.INFO
        and f"0 ack timeouts, {AUTO_LOCK_READ_FAILURE_THRESHOLD} response timeouts"
        in r.getMessage()
    ]
    assert len(latch) == 1


@pytest.mark.asyncio
async def test_auto_lock_read_value_in_flight_holds_without_strike() -> None:
    """Case 3 timing: within the response window the read waits, not strikes.

    Before the deadline the 0xBB may still be in flight, so the read neither
    books a response timeout nor issues another read.
    """
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:21", always_connected=True)
    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(return_value=None)

    assert await push_lock._read_auto_lock_setting(mock_lock) is True  # arms pending
    mock_lock.auto_lock_status.reset_mock()

    # Still inside the window (deadline is ~now + AUTO_LOCK_READ_RESPONSE_TIMEOUT).
    assert push_lock._auto_lock_response_deadline > time.monotonic()
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    mock_lock.auto_lock_status.assert_not_awaited()
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_read_response_failures == 0


@pytest.mark.asyncio
async def test_auto_lock_read_value_landing_during_ack_does_not_arm_pending() -> None:
    """A 0xBB that lands in the same loop turn as the ack is not waited on.

    If the notify path delivers the value before the read coroutine resumes,
    AutoLockState is already seen, so the read must not arm a pending-response
    deadline for a value already in hand -- otherwise the next cycle would book
    one spurious response timeout.
    """
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:2a", always_connected=True)

    def _ack_and_value(*_args: object) -> None:
        # The 0xBB is dispatched on the notify path before the await resumes.
        push_lock._update_any_state(
            [AutoLockState(mode=AutoLockMode.TIMER, duration=30)]
        )

    mock_lock = MagicMock()
    mock_lock.auto_lock_status = AsyncMock(side_effect=_ack_and_value)

    assert await push_lock._read_auto_lock_setting(mock_lock) is True
    assert push_lock._awaiting_auto_lock_response is False
    assert AutoLockState in push_lock._seen_this_session

    # A later cycle finds the value already seen, not a phantom pending read.
    assert await push_lock._read_auto_lock_setting(mock_lock) is False
    assert push_lock._auto_lock_read_response_failures == 0


@pytest.mark.asyncio
async def test_auto_lock_read_pending_survives_reconnect() -> None:
    """Dropout: a reconnect mid-read keeps the pending-response state.

    The hold must outlive the connection, so the pending flag, its deadline, and
    the response count all persist across the reconnect that clears the seen set.
    """
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:23", always_connected=True)
    deadline = time.monotonic() + AUTO_LOCK_READ_RESPONSE_TIMEOUT
    push_lock._awaiting_auto_lock_response = True
    push_lock._auto_lock_response_deadline = deadline
    push_lock._auto_lock_read_response_failures = 1
    push_lock._seen_this_session.add(AutoLockState)

    mock_lock = MagicMock()
    mock_lock.connect = AsyncMock()
    mock_lock.is_connected = True

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        await push_lock._ensure_connected()

    # The reconnect cleared _seen_this_session but preserved the pending state.
    assert AutoLockState not in push_lock._seen_this_session
    assert push_lock._awaiting_auto_lock_response is True
    assert push_lock._auto_lock_response_deadline == deadline
    assert push_lock._auto_lock_read_response_failures == 1
    push_lock._cancel_disconnect_timer()


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_auto_lock_read_success_ack_then_value(always_connected: bool) -> None:
    """Case 4: a full working lock. The ack arms the pending-response deadline;
    the 0xBB value landing afterwards clears it and arms the refresh timer."""
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:24", always_connected=always_connected
    )
    push_lock._lock_info = TEST_LOCK_INFO
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))

    before = time.monotonic()
    with patch.object(push_lock, "_ensure_connected", return_value=mock_lock):
        await push_lock._update()

    # The ack completed the read and armed both the pending flag and its
    # deadline; the value has not arrived yet. The flag is read into a typed
    # local so asserting it True does not narrow the later is-False check away.
    mock_lock.auto_lock_status.assert_awaited_once()
    armed: bool = push_lock._awaiting_auto_lock_response
    assert armed is True
    assert push_lock._auto_lock_response_deadline > before
    assert AutoLockState not in push_lock._seen_this_session

    # The 0xBB then lands on the notify path and clears the pending state.
    push_lock._update_any_state([AutoLockState(mode=AutoLockMode.TIMER, duration=30)])
    cleared: bool = push_lock._awaiting_auto_lock_response
    assert cleared is False
    assert AutoLockState in push_lock._seen_this_session
    assert (
        push_lock._next_auto_lock_read_time >= before + AUTO_LOCK_READ_REFRESH_INTERVAL
    )
    assert push_lock.auto_lock == AutoLockState(mode=AutoLockMode.TIMER, duration=30)
    push_lock._cancel_disconnect_timer()
    push_lock._cancel_keepalive_timer()


@pytest.mark.asyncio
async def test_update_any_state_auth_change_is_applied() -> None:
    """An AuthState change through _update_any_state updates the auth field.

    Covers the auth branch of _update_any_state, which sits directly beside the
    auto lock success block; the two share the "if lock_state.x != state" shape,
    so an inserted auto lock block anchors against the auth branch in the diff.
    """
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:29", always_connected=False)
    assert push_lock._get_current_state().auth is None

    push_lock._update_any_state([AuthState(successful=True)])

    assert push_lock.auth == AuthState(successful=True)


@pytest.mark.asyncio
async def test_auto_lock_read_response_backoff_survives_connect_on_demand(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Case 3 under connect-on-demand: the hold outlives each connection.

    A not-always-connected lock that acks the read but withholds the value
    idle-disconnects between adverts. Each advert reconnects (clearing the seen
    set), yet the pending state and response count carry across, so the response
    backoff still latches instead of the read repeating on every connection.
    """
    caplog.set_level(logging.INFO)
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:25", always_connected=False)

    async def _advert_connect() -> Lock:
        # A fresh advert-driven connection: was disconnected, now reconnects,
        # which clears _seen_this_session exactly as _update's connect does.
        push_lock._client = None
        lock = MagicMock()
        lock.connect = AsyncMock()
        lock.is_connected = True
        lock.auto_lock_status = AsyncMock(return_value=None)  # ack ok, no 0xBB
        with patch.object(push_lock, "_get_lock_instance", return_value=lock):
            return await push_lock._ensure_connected()

    # First connection: the read is issued and the pending deadline armed.
    client = await _advert_connect()
    assert await push_lock._read_auto_lock_setting(client) is True

    # Each later connection: the inter-advert gap lapsed the window, so the
    # withheld value books a response timeout that survived the reconnect.
    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
        client = await _advert_connect()
        assert push_lock._awaiting_auto_lock_response is True  # survived reconnect
        assert await push_lock._read_auto_lock_setting(client) is True
        assert push_lock._auto_lock_read_response_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    # The threshold connection finally arms the backoff.
    push_lock._auto_lock_response_deadline = time.monotonic() - 1.0
    client = await _advert_connect()
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(client) is False
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_ack_backoff_survives_connect_on_demand() -> None:
    """Case 2 under connect-on-demand: a lock silent to the read accumulates ack
    timeouts across reconnects and backs off, rather than being re-asked on
    every connection."""
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:26", always_connected=False)

    async def _advert_connect() -> Lock:
        push_lock._client = None
        lock = MagicMock()
        lock.connect = AsyncMock()
        lock.is_connected = True
        lock.auto_lock_status = AsyncMock(side_effect=TimeoutError)
        with patch.object(push_lock, "_get_lock_instance", return_value=lock):
            return await push_lock._ensure_connected()

    for expected in range(1, AUTO_LOCK_READ_FAILURE_THRESHOLD):
        client = await _advert_connect()
        assert await push_lock._read_auto_lock_setting(client) is False
        assert push_lock._auto_lock_read_ack_failures == expected
        assert push_lock._earliest_auto_lock_read_time == NEVER_TIME

    client = await _advert_connect()
    before = time.monotonic()
    assert await push_lock._read_auto_lock_setting(client) is False
    assert push_lock._auto_lock_read_ack_failures == 0
    assert (
        push_lock._earliest_auto_lock_read_time
        >= before + AUTO_LOCK_READ_FAILURE_BACKOFF
    )
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_auto_lock_read_connects_after_advertisement() -> None:
    """Connect-on-demand wiring: an advertisement drives the connect, then the
    read runs on that connection.

    The lock is disconnected; an advertisement arrives and schedules the update;
    the deferred update connects on demand and issues the auto lock read.
    """
    push_lock = _auto_lock_push_lock("aa:bb:cc:dd:ee:27", always_connected=False)
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))
    ble_device = BLEDevice(push_lock.address, "Test Lock", None)
    ad = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={YALE_MFR_ID: b"\x01"},
        platform_data=(),
        tx_power=0,
    )

    with patch.object(push_lock, "_get_lock_instance", return_value=mock_lock):
        # The advertisement schedules a connect-on-demand update.
        push_lock.update_advertisement(ble_device, ad)
        assert push_lock._cancel_deferred_update is not None
        # Drive the scheduled update to completion.
        push_lock._deferred_update()
        assert push_lock._update_task is not None
        await push_lock._update_task

    # The connect happened after the advertisement, and the read ran on it.
    mock_lock.connect.assert_awaited()
    mock_lock.auto_lock_status.assert_awaited_once()
    assert push_lock._awaiting_auto_lock_response is True
    push_lock._running = False
    push_lock._cancel_disconnect_timer()
    push_lock._cancel_keepalive_timer()


@pytest.mark.parametrize("always_connected", [False, True])
@pytest.mark.asyncio
async def test_dead_lock_read_not_reached_earlier_read_propagates(
    always_connected: bool,
) -> None:
    """Case 1: a dead lock answers nothing. The unguarded door read runs before
    the auto lock read in _update and its timeout propagates (the connection
    layer handles the dead lock), so the auto lock read is never reached and its
    counters stay clean.
    """
    push_lock = _auto_lock_push_lock(
        "aa:bb:cc:dd:ee:28", always_connected=always_connected
    )
    push_lock._lock_info = TEST_LOCK_INFO
    mock_lock = _auto_lock_update_lock(AsyncMock(return_value=None))
    mock_lock.door_status = AsyncMock(side_effect=TimeoutError)

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        pytest.raises(TimeoutError),
    ):
        await push_lock._update()

    mock_lock.auto_lock_status.assert_not_awaited()
    assert push_lock._auto_lock_read_ack_failures == 0
    assert push_lock._auto_lock_read_response_failures == 0
    assert push_lock._awaiting_auto_lock_response is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ResponseError("no confirmation"), TimeoutError("no confirmation")],
)
async def test_set_auto_lock_write_retries_twice_then_gives_up(
    error: Exception,
) -> None:
    """The write retries AUTO_LOCK_WRITE_ATTEMPTS times, not the default four.

    A stalled write surfaces as either a ResponseError or, when the settings
    response never lands, a TimeoutError; both are retryable, so the count holds
    for the actual field failure as well as the synthetic one.
    """
    push_lock = PushLock(
        address="aa:bb:cc:dd:ee:13",
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._running = True

    mock_lock = MagicMock()
    mock_lock.set_auto_lock = AsyncMock(side_effect=error)

    with (
        patch.object(push_lock, "_ensure_connected", return_value=mock_lock),
        patch.object(push_lock, "_async_handle_disconnected", new_callable=AsyncMock),
        patch("yalexs_ble.push.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(type(error)),
    ):
        await push_lock._set_auto_lock(AutoLockMode.TIMER, 30)

    assert mock_lock.set_auto_lock.await_count == AUTO_LOCK_WRITE_ATTEMPTS


@pytest.mark.asyncio
async def test_execute_lock_operation_success_stamps_complete_state() -> None:
    """A force_* returning True advances the state to the completed status.

    Drives lock() to completion: the transitional LOCKING is stamped, the
    op-response reports success, and the completed LOCKED status is applied.
    """
    push_lock = _operational_push_lock()
    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(return_value=True)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    mock_lock.force_lock.assert_awaited_once()
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_execute_lock_operation_failure_displays_jammed() -> None:
    """A force_* returning False (its op-response reported a failure) never
    stamps the completed status; the operation applies JAMMED once its window
    closes (the parser already logged the named cause and emitted JAMMED)."""
    push_lock = _operational_push_lock()
    mock_lock = MagicMock()
    mock_lock.force_unlock = AsyncMock(return_value=False)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.unlock()

    mock_lock.force_unlock.assert_awaited_once()
    assert push_lock.lock_status == LockStatus.JAMMED


# ---------------------------------------------------------------------------
# Stamp state only at write-success + operation-window filter + unlatch
# ---------------------------------------------------------------------------


def _operational_push_lock(address: str = "aa:bb:cc:dd:ee:20") -> PushLock:
    """A running lock with lock_info and advertisement data, ready to operate."""
    push_lock = PushLock(
        address=address,
        key="0800200c9a66",
        key_index=1,
        always_connected=False,
    )
    push_lock._name = "Test Lock"
    push_lock._lock_info = TEST_LOCK_INFO
    push_lock._running = True
    push_lock._advertisement_data = AdvertisementData(
        local_name="Test Lock",
        service_data={},
        service_uuids=[],
        rssi=-50,
        manufacturer_data={},
        platform_data=(),
        tx_power=0,
    )
    return push_lock


def _known_state(lock: LockStatus, door: DoorStatus = DoorStatus.CLOSED) -> LockState:
    return LockState(
        lock=lock,
        door=door,
        battery=None,
        auth=None,
        auto_lock=None,
        auto_lock_prev=None,
    )


@pytest.mark.asyncio
async def test_lock_stamps_transitional_only_at_write_success():
    """The LOCKING transitional is stamped only when the command write reaches
    the lock (write-success) -- never at issue time -- and exactly once."""
    push_lock = _operational_push_lock()
    order: list[tuple[str, LockStatus | None]] = []

    def cb(lock_state, lock_info, connection_info):
        order.append(("state", lock_state.lock))

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        order.append(("write_success", None))
        write_success_callback()
        return True

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    # No state was emitted before the write reached the lock.
    first_state_index = next(i for i, ev in enumerate(order) if ev[0] == "state")
    assert first_state_index > order.index(("write_success", None))
    # Exactly one LOCKING stamp, then LOCKED on completion.
    assert [ev for ev in order if ev == ("state", LockStatus.LOCKING)] == [
        ("state", LockStatus.LOCKING)
    ]
    assert order[-1] == ("state", LockStatus.LOCKED)
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_write_success_without_pending_state_stamps_nothing():
    """A write-success with no pending transitional stamps no state.

    Every gated operation arms a pending transitional before its write, so the
    stamp's guard never sees None on any in-repo path; this pins the guard's
    other arc. The window itself still opens: opening is unconditional at the
    write-success site.
    """
    push_lock = _operational_push_lock()
    events: list[LockState] = []
    push_lock.register_callback(
        lambda lock_state, lock_info, connection_info: events.append(lock_state)
    )
    assert push_lock._pending_op_state is None

    push_lock._operation_write_success()

    assert events == []
    assert push_lock._operation_window_open is True


@pytest.mark.asyncio
async def test_window_filters_foreign_settle_until_close():
    """While the operation window is open a foreign lock-status settle is
    dropped; once the window closes the same value is admitted."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    push_lock._operation_window_open = True
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status == LockStatus.UNLOCKED  # dropped mid-window

    push_lock._close_operation_window()
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status == LockStatus.LOCKED  # admitted after close


@pytest.mark.asyncio
async def test_window_admits_door_and_battery_members():
    """Door and battery members of a frame received mid-window still apply;
    only the lock status is filtered out."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    push_lock._operation_window_open = True

    battery = BatteryState(voltage=6.0, percentage=80)
    push_lock._update_any_state([DoorStatus.OPENED, battery, LockStatus.LOCKED])

    assert push_lock.door_status == DoorStatus.OPENED  # door applied
    assert push_lock.battery == battery  # battery applied
    assert push_lock.lock_status == LockStatus.UNLOCKED  # lock filtered


@pytest.mark.asyncio
async def test_early_error_before_write_leaves_no_window_no_unknown():
    """A retryable failure before the command write opens no window and never
    stamps UNKNOWN; the display keeps its prior value."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=DisconnectedError("dropped before write")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.LOCKED  # unchanged; no UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [OperationIncompleteError("no op-response"), UnlatchError("after write")],
)
async def test_nonretryable_after_write_stamps_unknown(exc):
    """The two non-retryable types raised after write-success (a transitional
    is on display with no result coming) close the window and stamp UNKNOWN."""
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        raise exc

    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(type(exc)),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNKNOWN


@pytest.mark.asyncio
async def test_nonretryable_before_write_success_stamps_no_unknown():
    """A non-retryable failure before write-success leaves the state alone.

    UNKNOWN answers a transitional left on display with no result coming. If
    the write never reached the lock, no transitional was stamped and the
    displayed state is still the truth, so the handler's window-open guard
    takes its other arc: no close, no UNKNOWN.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(
        side_effect=OperationIncompleteError("before write")
    )

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    assert push_lock._pending_op_state is None
    assert push_lock.lock_status == LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_failure_op_response_applies_jammed_after_window():
    """The parser emits JAMMED for our own failure op-response mid-window (the
    state callback runs before session resolves the future), so it is dropped;
    the success==False path re-applies JAMMED after closing the window.

    Frame origin (field_frames.md): the failure op-response
    bb0b001b...00001f0000 (byte[15]=0x1f MECH_POSITION) makes the Lock parser
    emit LockStatus.JAMMED.
    """
    push_lock = _operational_push_lock()
    events: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        events.append(lock_state.lock)

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()  # opens window, stamps LOCKING
        # Parser emission of the failure op-response lands inside our window.
        push_lock._state_callback([LockStatus.JAMMED])
        return False  # op-response byte[15] != 0

    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    # LOCKING at write-success; the mid-window JAMMED produced no event; JAMMED
    # applied once, only after the window closed.
    assert events == [LockStatus.LOCKING, LockStatus.JAMMED]
    assert push_lock.lock_status == LockStatus.JAMMED


@pytest.mark.asyncio
async def test_retry_restamps_at_write_success_without_unknown():
    """A retryable failure after write-success keeps the last transitional on
    display (never UNKNOWN); the next attempt re-sets the pending transitional
    and re-stamps at its own write-success."""
    push_lock = _operational_push_lock()
    events: list[LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        events.append(lock_state.lock)

    push_lock.register_callback(cb)

    pending_at_entry: list[LockStatus | None] = []
    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        pending_at_entry.append(push_lock._pending_op_state)
        write_success_callback()  # opens window, stamps LOCKING
        if attempts == 1:
            raise DisconnectedError("dropped before ack")
        return True

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
    ):
        await push_lock.lock()

    assert attempts == 2  # first attempt was retried
    # Each attempt re-set the pending transitional before its write.
    assert pending_at_entry == [LockStatus.LOCKING, LockStatus.LOCKING]
    # Never UNKNOWN; LOCKING once (the second stamp is a no-op -- display
    # already reads LOCKING) then LOCKED.
    assert LockStatus.UNKNOWN not in events
    assert events == [LockStatus.LOCKING, LockStatus.LOCKED]
    assert push_lock.lock_status == LockStatus.LOCKED


@pytest.mark.asyncio
async def test_unlatch_stamps_unlatching_then_unlocked():
    """The new public unlatch() maps to force_unlatch, stamping UNLATCHING at
    write-success and UNLOCKED on success: the op-response arrives when the
    latch has returned from its open dwell, so UNLATCHED is never the
    completed state."""
    push_lock = _operational_push_lock()
    order: list[str | LockStatus] = []

    def cb(lock_state, lock_info, connection_info):
        order.append(lock_state.lock)

    push_lock.register_callback(cb)

    mock_lock = MagicMock()

    async def force_unlatch(write_success_callback):
        order.append("write_success")
        write_success_callback()
        return True

    mock_lock.force_unlatch = force_unlatch

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.unlatch()

    assert order == ["write_success", LockStatus.UNLATCHING, LockStatus.UNLOCKED]
    assert push_lock.lock_status == LockStatus.UNLOCKED


@pytest.mark.asyncio
async def test_queued_operation_emits_no_transitional_until_dequeued() -> None:
    """A second operation queued on the operation lock stamps nothing.

    The flapping-bug regression: with op1 in flight (force_lock blocked on an
    Event), a second lock() is issued. While queued behind op1 on the operation
    lock it must emit NO transitional; op2's LOCKING appears only after op1
    completes. No existing test drives two serialized operations at once.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3a")
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    gate = asyncio.Event()
    calls = 0

    async def gated_force_lock(write_success_callback: Callable[[], None]) -> bool:
        nonlocal calls
        calls += 1
        write_success_callback()
        if calls == 1:
            await gate.wait()  # hold op1 open while op2 queues behind it
        return True

    mock_lock = MagicMock()
    mock_lock.force_lock = gated_force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
    ):
        op1 = asyncio.create_task(push_lock.lock())
        for _ in range(50):
            await asyncio.sleep(0)
            if emissions:
                break
        # op1 has stamped its single LOCKING at write-success.
        assert emissions == [LockStatus.LOCKING]

        op2 = asyncio.create_task(push_lock.lock())
        for _ in range(10):
            await asyncio.sleep(0)
        # op2 is queued on the operation lock: no transitional while queued.
        assert emissions == [LockStatus.LOCKING]

        gate.set()
        await op1
        await op2

    # op2's LOCKING appears only AFTER op1's completion state (LOCKED).
    assert emissions == [
        LockStatus.LOCKING,
        LockStatus.LOCKED,
        LockStatus.LOCKING,
        LockStatus.LOCKED,
    ]
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_window_filter_drops_lock_status_admits_door_and_battery() -> None:
    """Literal filter opened via write-success: even mid-window JAMMED is dropped.

    Distinct from the existing window tests (which set the flag directly and do
    not feed JAMMED): the window is opened through the real
    _operation_write_success path, and mid-window JAMMED is dropped with NO
    special-casing (the window check precedes the jam-hold logic), while door
    and battery members of the same frame still pass.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3b")

    # Open the window as write-success would: stamp the pending transitional.
    push_lock._pending_op_state = LockStatus.LOCKING
    push_lock._operation_write_success()
    assert push_lock.lock_status is LockStatus.LOCKING
    assert push_lock._operation_window_open is True

    # A foreign settle mid-window is dropped (literal filter).
    push_lock._update_any_state([LockStatus.LOCKED])
    assert push_lock.lock_status is LockStatus.LOCKING

    # Foreign jam evidence mid-window is dropped too (no special-casing), and it
    # arms no hold -- the window check comes before the jam-hold logic.
    push_lock._update_any_state([LockStatus.JAMMED])
    assert push_lock.lock_status is LockStatus.LOCKING
    assert push_lock._jammed_hold_deadline == NEVER_TIME

    # Door and battery members of a mid-window frame still pass (filtered
    # per-member by the callers, not by _admit_lock_status).
    push_lock._update_any_state([DoorStatus.OPENED])
    assert push_lock.door_status is DoorStatus.OPENED

    battery = BatteryState(voltage=6.0, percentage=80)
    push_lock._update_any_state([battery])
    assert push_lock.battery == battery


@pytest.mark.asyncio
async def test_early_disconnect_leaves_no_window_and_no_unknown() -> None:
    """A retryable disconnect before any write-success retries to exhaustion.

    Distinct from the existing early-error test (which checks only the final
    display): this pins the retry COUNT -- force_lock is re-sent exactly
    DEFAULT_ATTEMPTS times because the write never reached the lock -- and that
    the display never changed (no transitional, no UNKNOWN stamp).
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3c")
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=DisconnectedError("boom"))

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
        patch.object(push_lock, "_async_handle_disconnected", AsyncMock()),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        pytest.raises(DisconnectedError),
    ):
        await push_lock.lock()

    # Retried to exhaustion (each attempt re-sends because the write never
    # reached the lock)...
    assert mock_lock.force_lock.await_count == DEFAULT_ATTEMPTS
    # ...and the display never changed: no transitional, no UNKNOWN stamp.
    assert emissions == []
    assert push_lock.lock_status is LockStatus.UNKNOWN
    assert push_lock._operation_window_open is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_operation_incomplete_after_write_success_stamps_unknown_once() -> None:
    """OperationIncompleteError after write-success: exact emission order, no retry.

    Distinct from the existing non-retryable test (which checks only the final
    display): this pins the full emission SEQUENCE [LOCKING, UNKNOWN] and that
    the non-retryable error ran force_lock exactly once.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3d")
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    def force_lock_side_effect(write_success_callback: Callable[[], None]) -> bool:
        # The write reached the lock (window opens, LOCKING on display) but the
        # result never arrived.
        write_success_callback()
        raise OperationIncompleteError("no op-response")

    mock_lock = MagicMock()
    mock_lock.force_lock = AsyncMock(side_effect=force_lock_side_effect)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_schedule_future_update"),
        pytest.raises(OperationIncompleteError),
    ):
        await push_lock.lock()

    # Non-retryable: force_lock ran exactly once.
    assert mock_lock.force_lock.await_count == 1
    # LOCKING at write-success, then UNKNOWN when the result never came.
    assert emissions == [LockStatus.LOCKING, LockStatus.UNKNOWN]
    assert push_lock.lock_status is LockStatus.UNKNOWN
    assert push_lock._operation_window_open is False
    push_lock._cancel_disconnect_timer()


@pytest.mark.asyncio
async def test_poll_site_honours_the_operation_window_filter() -> None:
    """The poll path in _update passes through the same window filter as the
    notify path: with the operation window open, a lock status fetched by the
    poll is dropped in favour of the current display (the _admit_lock_status
    call in _update).

    Operations serialize on the operation lock, so a poll cannot actually run
    mid-window in normal flow; the site is load-bearing defense-in-depth -- the
    both-sites invariant that keeps a single-site patch from letting the other
    site bypass the filter -- so it is exercised by forcing the window open.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._operation_window_open = True

    mock_lock = MagicMock()
    mock_lock.lock_info = AsyncMock(return_value=TEST_LOCK_INFO)
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)
    mock_lock.door_status = AsyncMock(return_value=DoorStatus.CLOSED)
    mock_lock.battery = AsyncMock(return_value=None)
    mock_lock.auto_lock_status = AsyncMock()

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        final_state = await push_lock._update()

    # The poll fetched LOCKED, but the open window kept the held JAMMED.
    assert final_state.lock == LockStatus.JAMMED


@pytest.mark.asyncio
async def test_cancelled_mid_operation_closes_window_without_unknown() -> None:
    """A bare CancelledError mid-operation closes the window and re-raises.

    CancelledError is a BaseException, so the generic ``except Exception`` never
    sees it; without the dedicated arm the operation window would leak open and
    the transitional freeze on display, unhealable by polling. A cancel is not
    evidence the lock did or did not move, so -- unlike the non-retryable
    OperationIncompleteError path -- NO UNKNOWN is stamped; the transitional
    stays until the next poll settles it.
    """
    push_lock = _operational_push_lock()
    push_lock._lock_state = _known_state(LockStatus.UNLOCKED)
    emissions: list[LockStatus] = []
    push_lock.register_callback(lambda ls, li, ci: emissions.append(ls.lock))

    async def force_lock(write_success_callback: Callable[[], None]) -> bool:
        write_success_callback()  # opens window, stamps LOCKING
        raise asyncio.CancelledError

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        pytest.raises(asyncio.CancelledError),
    ):
        await push_lock.lock()

    assert push_lock._operation_window_open is False
    # No UNKNOWN stamp; the transitional persists until a poll heals it.
    assert LockStatus.UNKNOWN not in emissions
    assert push_lock.lock_status == LockStatus.LOCKING


# ---------------------------------------------------------------------------
# Hold JAMMED on display for 30s at both state-application sites
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_jammed_extends_hold_deadline():
    """Every admitted JAMMED event -- even one identical to the displayed
    value -- pushes the hold deadline out again (the helper runs before the
    equality check)."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:28")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status == LockStatus.JAMMED
        assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME

    # A second JAMMED while already displaying JAMMED still re-arms the hold.
    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock._jammed_hold_deadline == 1010.0 + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_extended_hold_pins_a_settle_past_the_first_deadline() -> None:
    """A repeated jam extends the hold, pinning a LOCKED settle past the FIRST end.

    Distinct from the existing extend test (which only checks the deadline
    value moved): this proves the extension actually PINS -- a LOCKED settle
    that falls past the original deadline but inside the extended one stays
    JAMMED, and only clears past the extended deadline.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3e")

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock._jammed_hold_deadline == 1000.0 + JAMMED_HOLD_TIME

    # Identical value -> no display change, but the deadline extends to 1050.
    with patch("yalexs_ble.push.time.monotonic", return_value=1020.0):
        push_lock._update_any_state([LockStatus.JAMMED])
        assert push_lock.lock_status is LockStatus.JAMMED
        assert push_lock._jammed_hold_deadline == 1020.0 + JAMMED_HOLD_TIME

    # Past the FIRST deadline (1030), inside the extended one.
    with patch("yalexs_ble.push.time.monotonic", return_value=1040.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status is LockStatus.JAMMED  # pinned by the extension

    with patch("yalexs_ble.push.time.monotonic", return_value=1051.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status is LockStatus.LOCKED


@pytest.mark.asyncio
async def test_admit_lock_status_poll_path_pins_and_extends() -> None:
    """The poll path (_admit_lock_status directly) pins JAMMED and extends on jam.

    Distinct entry path from the existing poll-site test (which drives the full
    _update merge): this unit-tests the shared _admit_lock_status helper, both
    the pin (a LOCKED poll answer inside the hold reads JAMMED) and the extend
    (a polled JAMMED arms a fresh deadline).
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:3f")
    # A hold is armed (e.g. by a settled 0x07 push or an op-response jam).
    push_lock._jammed_hold_deadline = 1000.0 + JAMMED_HOLD_TIME

    with patch("yalexs_ble.push.time.monotonic", return_value=1000.0):
        # A LOCKED poll answer inside the hold is pinned to JAMMED.
        assert (
            push_lock._admit_lock_status(LockStatus.LOCKED, LockStatus.JAMMED)
            is LockStatus.JAMMED
        )

    d0 = push_lock._jammed_hold_deadline
    with patch("yalexs_ble.push.time.monotonic", return_value=1005.0):
        # An incoming JAMMED from a poll answer arms/extends the deadline.
        assert (
            push_lock._admit_lock_status(LockStatus.JAMMED, LockStatus.LOCKED)
            is LockStatus.JAMMED
        )
    assert push_lock._jammed_hold_deadline > d0
    assert push_lock._jammed_hold_deadline == 1005.0 + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_hold_pins_locked_poll_then_admits_after_expiry():
    """Within the hold a LOCKED value is pinned to JAMMED; past the deadline it
    is admitted."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:29")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = 1030.0

    with patch("yalexs_ble.push.time.monotonic", return_value=1010.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status == LockStatus.JAMMED  # pinned within hold

    with patch("yalexs_ble.push.time.monotonic", return_value=1040.0):
        push_lock._update_any_state([LockStatus.LOCKED])
        assert push_lock.lock_status == LockStatus.LOCKED  # admitted after expiry


@pytest.mark.asyncio
async def test_write_success_releases_hold_and_shows_transitional():
    """A new operation's write-success releases the JAMMED hold FIRST, so its
    transitional passes the filter and is displayed (order: release hold ->
    stamp pending -> open window)."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2a")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = time.monotonic() + JAMMED_HOLD_TIME
    push_lock._pending_op_state = LockStatus.LOCKING

    push_lock._operation_write_success()

    assert push_lock._jammed_hold_deadline == NEVER_TIME  # hold released
    assert push_lock.lock_status == LockStatus.LOCKING  # transitional shown
    assert push_lock._operation_window_open is True


@pytest.mark.asyncio
async def test_poll_site_pins_jammed_via_update_merge():
    """A LOCKED status polled by _update while JAMMED is held is pinned to
    JAMMED at the merge site."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2b")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = time.monotonic() + JAMMED_HOLD_TIME
    # Only lock status is polled this cycle.
    push_lock._seen_this_session.update({BatteryState, DoorStatus, AutoLockState})

    mock_lock = MagicMock()
    mock_lock.lock_status = AsyncMock(return_value=LockStatus.LOCKED)

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        final_state = await push_lock._update()

    mock_lock.lock_status.assert_awaited_once()
    assert final_state.lock == LockStatus.JAMMED  # polled LOCKED pinned to JAMMED
    assert push_lock.lock_status == LockStatus.JAMMED


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06])
async def test_forced_reconnect_suppressed_during_jammed_hold(bad):
    """A polled 01/06 during a JAMMED hold is masked to JAMMED at the merge, so
    the forced reconnect is deliberately suppressed."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2c")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    push_lock._jammed_hold_deadline = time.monotonic() + JAMMED_HOLD_TIME
    push_lock._seen_this_session.update({BatteryState, DoorStatus, AutoLockState})

    mock_lock = MagicMock()
    mock_lock.lock_status = AsyncMock(return_value=bad)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_execute_forced_disconnect", AsyncMock()) as mock_disc,
    ):
        final_state = await push_lock._update()

    assert final_state.lock == LockStatus.JAMMED
    mock_disc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [LockStatus.UNKNOWN_01, LockStatus.UNKNOWN_06])
async def test_forced_reconnect_still_fires_without_hold(bad):
    """Without a JAMMED hold, a polled 01/06 still forces the reconnect
    (upstream behaviour outside the hold is unchanged)."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2d")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)  # no hold
    push_lock._seen_this_session.update({BatteryState, DoorStatus, AutoLockState})

    mock_lock = MagicMock()
    mock_lock.lock_status = AsyncMock(return_value=bad)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch.object(push_lock, "_execute_forced_disconnect", AsyncMock()) as mock_disc,
    ):
        final_state = await push_lock._update()

    assert final_state.lock == bad
    mock_disc.assert_awaited_once()


@pytest.mark.asyncio
async def test_operation_failure_arms_hold():
    """The success==False JAMMED (applied by the C3 path) also arms the display
    hold deadline."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2e")

    mock_lock = MagicMock()

    async def force_lock(write_success_callback):
        write_success_callback()
        return False  # op-response byte[15] != 0

    mock_lock.force_lock = force_lock

    before = time.monotonic()
    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._jammed_hold_deadline >= before + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_settled_jammed_frame_arms_hold():
    """A settled 0x07 push (parsed to JAMMED) with no operation in flight arms
    the display hold and shows JAMMED.

    Frame origin (field_frames.md): settled push bb02...07... (byte[8]=0x07
    STATICPOSITION) parses to LockStatus.JAMMED.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:2f")
    push_lock._lock_state = _known_state(LockStatus.LOCKED)

    before = time.monotonic()
    push_lock._state_callback([LockStatus.JAMMED])

    assert push_lock.lock_status == LockStatus.JAMMED
    assert push_lock._jammed_hold_deadline >= before + JAMMED_HOLD_TIME


@pytest.mark.asyncio
async def test_battery_only_cycle_does_not_move_hold_deadline():
    """CONFORMANCE FIX: a cycle that polls battery but not lock status must not
    touch the JAMMED hold deadline -- a carried-forward cached JAMMED is not
    fresh evidence and must not re-arm the hold."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:30")
    push_lock._lock_state = _known_state(LockStatus.JAMMED)
    deadline = time.monotonic() + JAMMED_HOLD_TIME
    push_lock._jammed_hold_deadline = deadline
    # Lock status already seen (skipped; not always_connected); battery due.
    push_lock._seen_this_session.update({LockStatus, DoorStatus, AutoLockState})

    mock_lock = MagicMock()
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        final_state = await push_lock._update()

    mock_lock.battery.assert_awaited_once()
    mock_lock.lock_status.assert_not_called()
    # Deadline untouched; display still JAMMED (carried forward, not re-armed).
    assert push_lock._jammed_hold_deadline == deadline
    assert final_state.lock == LockStatus.JAMMED


# ---------------------------------------------------------------------------
# Extend the stale-state debounce to the op-response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ack_and_op_response_callbacks_stamp_their_anchors():
    """_ack_callback stamps only the acknowledgement anchor and
    _op_response_callback only the completion anchor; an external op (no command
    on our side) stamps neither the start nor the acknowledgement anchor."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:32")
    assert push_lock._last_lock_operation_start_time == NEVER_TIME
    assert push_lock._last_lock_operation_acknowledged_time == NEVER_TIME
    assert push_lock._last_lock_operation_complete_time == NEVER_TIME

    with patch("yalexs_ble.push.time.monotonic", return_value=2000.0):
        push_lock._ack_callback()
    assert push_lock._last_lock_operation_acknowledged_time == 2000.0
    assert push_lock._last_lock_operation_start_time == NEVER_TIME  # not the command

    with patch("yalexs_ble.push.time.monotonic", return_value=2005.0):
        push_lock._op_response_callback()
    assert push_lock._last_lock_operation_complete_time == 2005.0
    # An external op-response stamps neither the start nor the ack anchor.
    assert push_lock._last_lock_operation_start_time == NEVER_TIME
    assert push_lock._last_lock_operation_acknowledged_time == 2000.0


@pytest.mark.asyncio
async def test_get_lock_instance_wires_only_the_stream_observers():
    """_get_lock_instance stores the two stream observers and nothing else.

    The acknowledgement and op-response hooks fire on any matching frame,
    including operations we did not issue, so they belong to the connection.
    The write-success hook belongs to a single operation and is handed over at
    the call site instead, so a Lock built here carries no way to open the
    operation window on its own.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:33")
    push_lock._ble_device = MagicMock()

    lock = push_lock._get_lock_instance()

    assert lock._ack_callback == push_lock._ack_callback
    assert lock._op_response_callback == push_lock._op_response_callback
    assert not hasattr(lock, "_write_success_callback")


@pytest.mark.asyncio
async def test_operation_outside_the_gate_cannot_open_the_window():
    """A force_* issued straight at the Lock leaves the operation window shut.

    Only _execute_lock_operation hands over the write-success hook, and only
    that method closes the window again. A caller reaching the Lock directly
    therefore gets no hook and cannot leave a window open with no operation in
    flight, which would freeze status admission for the object's life.
    """
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:34")
    push_lock._ble_device = MagicMock()
    lock = push_lock._get_lock_instance()
    lock.session = MagicMock()
    lock.secure_session = MagicMock()
    lock.client = MagicMock(is_connected=True)

    handed: list[object] = []

    async def _capture(
        command: bytearray,
        command_name: str,
        response_timeout: float | None = None,
        progress: object | None = None,
        write_success_callback: Callable[[], None] | None = None,
    ) -> bool:
        handed.append(write_success_callback)
        if write_success_callback is not None:
            write_success_callback()
        return True

    lock._execute_operation_command = _capture  # type: ignore[method-assign]

    assert await lock.force_lock() is True

    assert handed == [None]
    assert push_lock._operation_window_open is False


@pytest.mark.asyncio
async def test_execute_lock_operation_hands_the_hook_to_the_operation():
    """The gated path passes its own write-success hook to the operation."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:35")
    received: list[object] = []

    async def force_lock(write_success_callback):
        received.append(write_success_callback)
        return True

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with patch.object(
        push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)
    ):
        await push_lock.lock()

    assert received == [push_lock._operation_write_success]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start", "ack", "complete", "expected_delay"),
    [
        # Acknowledgement arm dominates: max(start, ack) + 6.1.
        (996.0, 997.0, 990.0, 3.1),
        # Command-issue arm dominates over a stale acknowledgement.
        (998.0, 995.0, 990.0, 4.1),
        # Op-response arm dominates: complete + 4.1.
        (990.0, 991.0, 999.0, 3.1),
    ],
)
async def test_deferred_update_reschedules_to_the_max_anchor_arm(
    start, ack, complete, expected_delay
):
    """Within the stale window the update reschedules by exactly
    max(max(start, ack) + 6.1, complete + 4.1) - now."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:34")
    push_lock._last_lock_operation_start_time = start
    push_lock._last_lock_operation_acknowledged_time = ack
    push_lock._last_lock_operation_complete_time = complete

    with (
        patch("yalexs_ble.push.time.monotonic", return_value=1000.0),
        patch.object(
            push_lock, "_schedule_future_update_with_debounce"
        ) as mock_reschedule,
    ):
        push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert mock_reschedule.call_args.args[0] == pytest.approx(expected_delay)


@pytest.mark.asyncio
async def test_external_op_response_alone_defers_poll():
    """An external op-response (op_response_callback, with the start and
    acknowledgement anchors never stamped) alone defers the next poll by 4.1 s."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:35")
    # A fake "now" comfortably after NEVER_TIME so the untouched start/ack
    # anchors (still NEVER_TIME) stay firmly in the past.
    now = NEVER_TIME + 90000.0

    with patch("yalexs_ble.push.time.monotonic", return_value=now):
        push_lock._op_response_callback()
        # The external op produced no start or acknowledgement anchor.
        assert push_lock._last_lock_operation_start_time == NEVER_TIME
        assert push_lock._last_lock_operation_acknowledged_time == NEVER_TIME
        with patch.object(
            push_lock, "_schedule_future_update_with_debounce"
        ) as mock_reschedule:
            push_lock._deferred_update()

    mock_reschedule.assert_called_once()
    assert mock_reschedule.call_args.args[0] == pytest.approx(
        POST_OP_RESPONSE_DEBOUNCE_DELAY
    )


@pytest.mark.asyncio
async def test_no_anchors_proceeds_immediately():
    """With every anchor at NEVER_TIME (about a day in the past) the deadline is
    far behind now, so _deferred_update starts the update instead of deferring."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:36")
    assert push_lock._last_lock_operation_start_time == NEVER_TIME
    assert push_lock._last_lock_operation_acknowledged_time == NEVER_TIME
    assert push_lock._last_lock_operation_complete_time == NEVER_TIME

    push_lock._execute_deferred_update = AsyncMock()  # type: ignore[method-assign]

    with patch.object(
        push_lock, "_schedule_future_update_with_debounce"
    ) as mock_reschedule:
        push_lock._deferred_update()
        assert push_lock._update_task is not None
        await push_lock._update_task

    mock_reschedule.assert_not_called()


@pytest.mark.asyncio
async def test_execute_operation_restamps_start_anchor_per_attempt():
    """The start anchor is stamped immediately before each operation attempt, so
    a retry re-stamps it and the stale-state floor tracks the latest attempt."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:37")
    start_stamps: list[float] = []
    attempts = 0

    async def force_lock(write_success_callback):
        nonlocal attempts
        attempts += 1
        start_stamps.append(push_lock._last_lock_operation_start_time)
        if attempts == 1:
            raise DisconnectedError("dropped before write")
        return True

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock
    monotonic = itertools.count(1000.0, 1.0)

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.asyncio.sleep", AsyncMock()),
        patch("yalexs_ble.push.time.monotonic", side_effect=monotonic),
    ):
        await push_lock.lock()

    assert attempts == 2
    # Each attempt stamped a start anchor; the retry re-stamped it strictly
    # later, so the final anchor is the second attempt's.
    assert len(start_stamps) == 2
    assert start_stamps[1] > start_stamps[0]
    assert push_lock._last_lock_operation_start_time == start_stamps[1]


# ---------------------------------------------------------------------------
# Defer battery polls past the post-operation voltage sag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [True, False], ids=["success", "jam"])
async def test_operation_defers_battery_poll_past_voltage_sag(outcome):
    """Both operation outcomes (success and jam) arm the post-op battery
    cooldown; _poll_battery then skips within the window and polls past it."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:38")
    # Base "now" past NEVER_TIME so the cooldown arithmetic is not masked by the
    # NEVER_TIME sentinel default of _earliest_battery_attempt_time.
    base = NEVER_TIME + 90000.0

    async def force_lock(write_success_callback):
        write_success_callback()
        return outcome

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock
    mock_lock.battery = AsyncMock(return_value=BatteryState(voltage=6.0, percentage=80))

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.time.monotonic", return_value=base),
    ):
        await push_lock.lock()

    # The operation armed the post-op cooldown regardless of outcome.
    assert push_lock._earliest_battery_attempt_time == (
        base + POST_OPERATION_BATTERY_COOLDOWN
    )
    # Drop the RESYNC poll the state change scheduled; this test drives
    # _poll_battery directly.
    push_lock._cancel_future_update()

    state = _known_state(LockStatus.LOCKED)
    # Inside the cooldown window: the poll is skipped (would sample the sag).
    with patch("yalexs_ble.push.time.monotonic", return_value=base + 29.0):
        _, made_request = await push_lock._poll_battery(mock_lock, state)
    assert made_request is False
    mock_lock.battery.assert_not_called()

    # Past the window: the poll proceeds.
    with patch("yalexs_ble.push.time.monotonic", return_value=base + 31.0):
        _, made_request = await push_lock._poll_battery(mock_lock, state)
    assert made_request is True
    mock_lock.battery.assert_awaited_once()


@pytest.mark.asyncio
async def test_operation_does_not_shorten_a_longer_battery_cooldown():
    """A live battery-timeout cooldown further out than the 30 s post-op window
    is preserved (max), never shortened by an operation."""
    push_lock = _operational_push_lock("aa:bb:cc:dd:ee:39")
    base = NEVER_TIME + 90000.0
    # A battery timeout already set a cooldown far past the post-op window.
    longer_cooldown = base + BATTERY_TIMEOUT_COOLDOWN
    push_lock._earliest_battery_attempt_time = longer_cooldown

    async def force_lock(write_success_callback):
        write_success_callback()
        return True

    mock_lock = MagicMock()
    mock_lock.force_lock = force_lock

    with (
        patch.object(push_lock, "_ensure_connected", AsyncMock(return_value=mock_lock)),
        patch("yalexs_ble.push.time.monotonic", return_value=base),
    ):
        await push_lock.lock()

    # max() keeps the longer live cooldown; base + 30 would have shortened it.
    assert push_lock._earliest_battery_attempt_time == longer_cooldown

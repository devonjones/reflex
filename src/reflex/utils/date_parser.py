"""Date parsing utilities for snooze functionality."""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from dateutil import parser as dateutil_parser


def parse_snooze_date(date_str: str, reference_time: Optional[datetime] = None) -> Optional[datetime]:
    """Parse a snooze date string into a datetime.

    Supports formats:
    - Relative: '3d', '1w', '2w', 'tomorrow'
    - Natural: 'next week', 'next monday'
    - Absolute: 'jan 20', 'january 20', '2026-01-20'

    Args:
        date_str: Date string to parse
        reference_time: Reference time for relative dates (defaults to now)

    Returns:
        Parsed datetime with timezone, or None if parse fails
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    date_str = date_str.strip().lower()

    # Handle relative formats: 3d, 1w, 2w
    relative_match = re.match(r'^(\d+)\s*([dw])$', date_str)
    if relative_match:
        amount = int(relative_match.group(1))
        unit = relative_match.group(2)

        if unit == 'd':
            return reference_time + timedelta(days=amount)
        elif unit == 'w':
            return reference_time + timedelta(weeks=amount)

    # Handle 'tomorrow'
    if date_str == 'tomorrow':
        return reference_time + timedelta(days=1)

    # Handle 'next week'
    if date_str == 'next week':
        return reference_time + timedelta(weeks=1)

    # Handle 'next monday', 'next tuesday', etc.
    weekday_match = re.match(r'^next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$', date_str)
    if weekday_match:
        weekday_name = weekday_match.group(1)
        weekday_map = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6,
        }
        target_weekday = weekday_map[weekday_name]
        current_weekday = reference_time.weekday()

        # Calculate days until next occurrence
        days_ahead = target_weekday - current_weekday
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7

        return reference_time + timedelta(days=days_ahead)

    # Try dateutil parser for absolute dates
    try:
        # Parse with dateutil (handles many formats)
        parsed = dateutil_parser.parse(date_str, default=reference_time)

        # Ensure timezone aware
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed
    except (ValueError, dateutil_parser.ParserError):
        return None

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from curioslm.checkpoints import stale_orphans

NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def lfs(oid, minutes_ago):
    return SimpleNamespace(file_oid=oid, pushed_at=NOW - timedelta(minutes=minutes_ago))


def test_prunes_only_old_versions_nothing_points_to():
    files = [
        lfs("current-a0", 90),       # still the live checkpoint: keep
        lfs("old-a0", 60),           # replaced long ago: prune
        lfs("just-uploaded", 1),     # unreferenced but seconds old, e.g. another run's pending commit: keep
    ]

    pruned = stale_orphans(files, current_oids={"current-a0"}, older_than=NOW - timedelta(minutes=10))

    assert [f.file_oid for f in pruned] == ["old-a0"]

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread, current_thread


class PetStore:
    """Read and update the DigiPet state stored in SQLite."""

    PET_ID = 1
    CARE_AMOUNT = 30
    DECAY_INTERVAL_SECONDS = 10

    @dataclass(frozen=True)
    class Stats:
        """A snapshot of the pet state returned by the store."""

        name: str
        hunger: int
        thirst: int
        boredom: int
        updated_at: str

        @property
        def mood(self):
            """Derive the pet's mood from its most urgent need."""
            highest_need = max(self.hunger, self.thirst, self.boredom)
            if highest_need <= 25:
                return "happy"
            if highest_need <= 50:
                return "okay"
            if highest_need <= 75:
                return "needs care"
            return "struggling"

        def to_dict(self):
            """Return JSON-ready pet statistics."""
            return {
                "name": self.name,
                "hunger": self.hunger,
                "thirst": self.thirst,
                "boredom": self.boredom,
                "updated_at": self.updated_at,
                "mood": self.mood,
            }

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._stop_event = Event()
        self._timer_thread = Thread(
            target=self._run_decay_timer,
            name="digipet-stat-decay",
            daemon=True,
        )
        self._timer_thread.start()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @classmethod
    def _stats_from_row(cls, row):
        if row is None:
            raise RuntimeError(f"Pet with id {cls.PET_ID} was not found")

        return cls.Stats(
            name=row["name"],
            hunger=row["hunger"],
            thirst=row["thirst"],
            boredom=row["boredom"],
            updated_at=row["updated_at"],
        )

    def get_stats(self):
        """Return the current pet state."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT name, hunger, thirst, boredom, updated_at
                FROM pet
                WHERE id = ?
                """,
                (self.PET_ID,),
            ).fetchone()

        return self._stats_from_row(row)

    def _run_decay_timer(self):
        while not self._stop_event.wait(self.DECAY_INTERVAL_SECONDS):
            self._decay_stats()

    def _decay_stats(self):
        updated_at = datetime.now(timezone.utc).isoformat()

        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE pet
                    SET hunger = MAX(0, hunger - 2),
                        thirst = MAX(0, thirst - 2),
                        boredom = MAX(0, boredom - 3),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (updated_at, self.PET_ID),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Pet with id {self.PET_ID} was not found")

    def close(self):
        """Stop the automatic stat-decay timer."""
        self._stop_event.set()
        if current_thread() is not self._timer_thread:
            self._timer_thread.join()

    def _care_for(self, need):
        if need not in {"hunger", "thirst", "boredom"}:
            raise ValueError(f"Unknown pet need: {need}")

        updated_at = datetime.now(timezone.utc).isoformat()

        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    f"""
                    UPDATE pet
                    SET {need} = MIN(100, {need} + ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (self.CARE_AMOUNT, updated_at, self.PET_ID),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"Pet with id {self.PET_ID} was not found")

                row = connection.execute(
                    """
                    SELECT name, hunger, thirst, boredom, updated_at
                    FROM pet
                    WHERE id = ?
                    """,
                    (self.PET_ID,),
                ).fetchone()

        return self._stats_from_row(row)

    def feed(self):
        """Feed the pet and return its updated state."""
        return self._care_for("hunger")

    def give_water(self):
        """Give the pet water and return its updated state."""
        return self._care_for("thirst")

    def play(self):
        """Play with the pet and return its updated state."""
        return self._care_for("boredom")

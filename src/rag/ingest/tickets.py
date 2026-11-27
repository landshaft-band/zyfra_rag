import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


TICKET_HEADING = re.compile(
    r"^## (?P<ticket_id>DD-[0-9]+) — (?P<title>.+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    title: str
    body: str


def iter_tickets(path: str | Path) -> Iterator[Ticket]:
    """Yield tickets in the order they appear in one Markdown export."""
    markdown = Path(path).read_text(encoding="utf-8")
    headings = TICKET_HEADING.finditer(markdown)
    heading = next(headings, None)

    while heading is not None:
        next_heading = next(headings, None)
        body_end = next_heading.start() if next_heading is not None else len(markdown)
        body = markdown[heading.end():body_end].strip("\n")
        body = body.removesuffix("\n\n---")

        yield Ticket(
            ticket_id=heading["ticket_id"],
            title=heading["title"].strip(),
            body=body,
        )
        heading = next_heading


def read_ticket(path: str | Path, ticket_id: str) -> Ticket:
    """Read one ticket from the Markdown export by its exact ID."""
    for ticket in iter_tickets(path):
        if ticket.ticket_id == ticket_id:
            return ticket
    raise ValueError(f"Ticket {ticket_id!r} was not found in {path}")

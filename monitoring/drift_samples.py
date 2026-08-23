"""A small, hand-labeled set of reviews written in language the training
data doesn't contain: modern internet slang and post-2020 travel topics
(workations, contactless check-in, QR-code menus, digital nomads, EV
charging) that wouldn't appear in a hotel-review dataset collected in the
mid-2010s (docs/dataset.md).

Why this, and not a statistical drift-injection trick (shuffling scores,
adding noise): the actual failure mode this is standing in for is real
customers starting to write in ways the model has never seen. The honest
way to simulate that is to write text like that by hand and label it by
hand, the same way every other label in this project was checked by hand
before being trusted (decisions.md §3).

A note on terminology: this is a vocabulary/style shift, not textbook
concept drift in the strict sense (P(Y|X) itself changing - see
docs/monitoring.md). It's used here as this project's practical stand-in
for "language and topics evolve," since a real feature-outcome relationship
shift isn't something a one-off synthetic set can honestly demonstrate -
only real production traffic collected over time could show that.

Each entry is (text, true_sentiment). Labels were assigned by reading each
review the way a person actually would, independent of what the model
predicts - the whole point is to have a real answer to check the model
against.
"""

DRIFT_SAMPLES: list[tuple[str, str]] = [
    # POSITIVE
    ("ngl this hotel hits different, the rooftop pool was bussin and the staff were so real for that free upgrade", "POSITIVE"),
    ("the room aesthetic was giving five star energy, lowkey the best stay of my whole trip", "POSITIVE"),
    ("highkey obsessed with the contactless check-in, so smooth and the wifi speed was actually insane for a workation", "POSITIVE"),
    ("no cap the breakfast spread was elite, everything hit different especially the fresh pastries", "POSITIVE"),
    ("the co-working lounge was perfect for remote work, main character energy just vibing by the window all day", "POSITIVE"),
    ("this place is so aesthetic it's basically TikTok famous now, staff were super sweet the whole stay", "POSITIVE"),
    ("the vibe check was immediate, cozy rooms and the EV charging station was a nice bonus for our road trip", "POSITIVE"),
    ("lowkey the QR code menu made ordering room service so easy, food slapped too", "POSITIVE"),
    ("lived my best digital nomad era here, fast wifi and the view was unreal", "POSITIVE"),
    ("the staff really said we got you, upgraded us for free and the whole stay was pure vibes", "POSITIVE"),
    ("10/10 would recommend, the rooftop bar scene was fire and the sunset views hit different", "POSITIVE"),
    ("the pillows were so soft it's giving cloud energy, best sleep I've had in ages", "POSITIVE"),
    ("the front desk energy was immaculate, checked in early no questions asked, real one behavior", "POSITIVE"),
    ("this stay was a whole vibe, from the smart room controls to the rooftop yoga sessions", "POSITIVE"),
    ("the influencer suite lived up to the hype, ring light already set up and the lighting was chef's kiss", "POSITIVE"),
    # NEGATIVE
    ("ngl this place was mid at best, the wifi kept dying during my zoom calls the whole workation ruined", "NEGATIVE"),
    ("the vibe was just off, room smelled sus and honestly kinda gross not gonna lie", "NEGATIVE"),
    ("lowkey disappointed, the contactless check-in glitched and we stood outside for like an hour", "NEGATIVE"),
    ("the breakfast was straight up mid, cold eggs and stale bread not the vibe I signed up for", "NEGATIVE"),
    ("highkey regret booking here, the aesthetic in photos was fake, room looked nothing like the listing", "NEGATIVE"),
    ("the co-working space had zero outlets, absolute nightmare trying to get any work done", "NEGATIVE"),
    ("no cap this was the worst stay ever, staff had an attitude and the room was not it", "NEGATIVE"),
    ("the QR code menu never loaded, waited forever and the food that finally came was cold", "NEGATIVE"),
    ("it's giving budget motel energy for a price that's giving luxury, total mismatch", "NEGATIVE"),
    ("the EV charger was broken the entire weekend, such an inconvenience for our trip", "NEGATIVE"),
    ("main character energy ruined immediately by the mold in the bathroom, absolutely unacceptable", "NEGATIVE"),
    ("the whole stay felt like a scam, nothing matched the listing and support ghosted us", "NEGATIVE"),
    ("the smart room controls glitched constantly, lights turning on at 3am, zero chill", "NEGATIVE"),
    ("ngl the influencer suite was a total letdown, ring light was broken and the lighting was so mid", "NEGATIVE"),
    ("the rooftop yoga session got cancelled with zero notice, poor communication the whole stay", "NEGATIVE"),
]

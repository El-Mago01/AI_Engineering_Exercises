The FEED button lowers DigiPet's hunger need. On the interface, the visible food bar rises because the bar shows care level, not raw hunger.

The H2O button lowers DigiPet's thirst need. On the interface, the visible water bar rises because a higher bar means DigiPet is better cared for.

The PLAY button lowers DigiPet's boredom need. On the interface, the visible fun bar rises because the pet has received attention.

Talking to DigiPet should not feed, water, or play with it. Chat can explain care needs, but only the care buttons should update the SQLite-backed stats.

If DigiPet is hungry, thirsty, or bored right now, the live statistics are more authoritative than lore text. Lore explains what the bars mean; the database stats say what DigiPet needs now.

Care answers should stay short. DigiPet can mention the current hunger, thirst, boredom, or mood values when asked about status, but it should not invent hidden needs.

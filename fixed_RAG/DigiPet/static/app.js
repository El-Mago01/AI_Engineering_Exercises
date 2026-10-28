/**
 * DigiPet UI — live stats from the server (DB). No manual refresh needed.
 */

const $ = (id) => document.getElementById(id);

const els = {
  lcd: $("lcd-screen"),
  sprite: $("pet-sprite"),
  name: $("pet-name"),
  mood: $("pet-mood"),
  careToast: $("care-toast"),
  bubble: $("pet-bubble"),
  talkForm: $("talk-form"),
  talkInput: $("talk-input"),
  talkBtn: $("btn-talk"),
  hunger: { bar: $("bar-hunger"), val: $("val-hunger") },
  thirst: { bar: $("bar-thirst"), val: $("val-thirst") },
  boredom: { bar: $("bar-boredom"), val: $("val-boredom") },
  status: $("status"),
  feed: $("btn-feed"),
  water: $("btn-water"),
  play: $("btn-play"),
};

const MOOD_EMOJI = {
  happy: "😄",
  okay: "🙂",
  "needs care": "😕",
  struggling: "😫",
};

const ACTION_EMOJI = {
  "/api/feed": "🍖",
  "/api/give-water": "🥤",
  "/api/play-with": "⚽",
};

const ACTION_SPRITE_MS = 1800;
const CHAT_PLACEHOLDER = "> press TALK when ready_";

let actionSprite = { emoji: null, until: 0, kind: null };
let chatHasReply = false;

const BUBBLE_IDLE = {
  happy: "> *happy beep*",
  okay: "> *content whistle*",
  "needs care": "> *restless shuffle*",
  struggling: "> *weak peep*",
};

function padStat(n) {
  return String(Math.round(n)).padStart(2, "0");
}

function setBar(barEl, value) {
  barEl.style.width = `${Math.min(100, Math.max(0, value))}%`;
}

function careLevel(needValue) {
  return 100 - Math.min(100, Math.max(0, needValue));
}

// Poll occasionally so the visible care levels stay fresh without flooding the terminal.
const SYNC_MS = 10000;

let syncInFlight = false;
let actionInFlight = false;
let talkInFlight = false;

function isActionSpriteActive() {
  return Boolean(actionSprite.emoji && Date.now() < actionSprite.until);
}

function applySpriteForMood(mood) {
  if (isActionSpriteActive()) {
    els.sprite.textContent = actionSprite.emoji;
    if (actionSprite.kind) {
      els.sprite.dataset.action = actionSprite.kind;
    }
    return;
  }
  els.sprite.textContent = MOOD_EMOJI[mood] || "🐣";
  delete els.sprite.dataset.action;
}

function startActionSprite(path) {
  const kind = path.replace("/api/", "");
  actionSprite = {
    emoji: ACTION_EMOJI[path] || "✨",
    until: Date.now() + ACTION_SPRITE_MS,
    kind,
  };
  els.sprite.textContent = actionSprite.emoji;
  els.sprite.dataset.action = kind;
  bouncePet();
}

function setChatBubble(text, { thinking = false } = {}) {
  els.bubble.textContent = text.startsWith(">") ? text : `> ${text}`;
  els.bubble.classList.toggle("thinking", thinking);
}

function setCareToast(text) {
  els.careToast.textContent = text || "";
}

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = "status" + (kind ? ` ${kind}` : "");
}

function renderStats(stats) {
  els.name.textContent = stats.name || "DigiPet";
  els.mood.textContent = stats.mood || "unknown";

  const mood = stats.mood || "okay";
  applySpriteForMood(mood);

  if (!chatHasReply && !talkInFlight) {
    setChatBubble(BUBBLE_IDLE[mood] || CHAT_PLACEHOLDER);
  }

  els.lcd.classList.toggle("mood-struggling", mood === "struggling");

  const food = careLevel(stats.hunger);
  const water = careLevel(stats.thirst);
  const fun = careLevel(stats.boredom);

  setBar(els.hunger.bar, food);
  setBar(els.thirst.bar, water);
  setBar(els.boredom.bar, fun);
  els.hunger.val.textContent = padStat(food);
  els.thirst.val.textContent = padStat(water);
  els.boredom.val.textContent = padStat(fun);
}

async function fetchStats() {
  const res = await fetch("/api/stats", { cache: "no-store" });
  if (!res.ok) throw new Error(`Stats failed (${res.status})`);
  return res.json();
}

async function postAction(path) {
  const res = await fetch(path, { method: "POST", cache: "no-store" });
  if (!res.ok) throw new Error(`Action failed (${res.status})`);
  return res.json();
}

async function postTalk(message) {
  const res = await fetch("/api/talk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error(`Talk failed (${res.status})`);
  return res.json();
}

function bouncePet() {
  els.sprite.classList.remove("bounce");
  void els.sprite.offsetWidth;
  els.sprite.classList.add("bounce");
}

async function syncStats() {
  if (syncInFlight || actionInFlight || talkInFlight) return;
  syncInFlight = true;
  try {
    const stats = await fetchStats();
    renderStats(stats);
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    syncInFlight = false;
  }
}

async function doAction(path, busyToast) {
  actionInFlight = true;
  startActionSprite(path);
  setButtonsDisabled(true);
  setCareToast(busyToast.replace(/^>\s*/, ""));
  try {
    const data = await postAction(path);
    actionSprite.until = Date.now() + ACTION_SPRITE_MS;
    renderStats(data.stats);
    bouncePet();
    setCareToast(data.message || "Done!");
    setStatus("Saved to database.", "ok");
  } catch (err) {
    setStatus(err.message, "error");
    actionSprite = { emoji: null, until: 0, kind: null };
  } finally {
    actionInFlight = false;
    setButtonsDisabled(false);
    setTimeout(() => setCareToast(""), 2500);
  }
}

async function doTalk() {
  const message = els.talkInput.value.trim();
  if (!message) {
    els.talkInput.focus();
    return;
  }

  talkInFlight = true;
  chatHasReply = true;
  setButtonsDisabled(true);
  setChatBubble("> ...", { thinking: true });

  try {
    const data = await postTalk(message);
    setChatBubble(`> ${data.reply}`);
    els.talkInput.value = "";
    setStatus("● TALK OK", "ok");
  } catch (err) {
    chatHasReply = false;
    setChatBubble(CHAT_PLACEHOLDER);
    setStatus(err.message, "error");
  } finally {
    talkInFlight = false;
    setButtonsDisabled(false);
    els.bubble.classList.remove("thinking");
  }
}

function setButtonsDisabled(disabled) {
  [els.feed, els.water, els.play, els.talkBtn].forEach((b) => {
    b.disabled = disabled;
  });
  els.talkInput.disabled = disabled;
}

els.feed.addEventListener("click", () => doAction("/api/feed", "> *munch munch*"));
els.water.addEventListener("click", () => doAction("/api/give-water", "> *slurp*"));
els.play.addEventListener("click", () => doAction("/api/play-with", "> *zoomies*"));

els.talkForm.addEventListener("submit", (e) => {
  e.preventDefault();
  doTalk();
});

syncStats().then(() => setStatus("● LIVE LINK OK", "ok"));

setInterval(() => {
  if (!document.hidden) {
    syncStats();
  }
}, SYNC_MS);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) syncStats();
});

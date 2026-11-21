# Reproducible Execution Evidence

Real output captured by running each command from the project root on Python 3.12.4. Regenerate any block by re-running its command; the recommendations are deterministic (the optional Gemini explanation is off in these runs). The full evaluation table is in [evaluation_results.md](evaluation_results.md) (`python -m src.evaluate`).

## `python -m src.main --ask "something calm for studying"`

```
Loaded songs: 20

You asked for: "something calm for studying"

What I looked up:
  - Studying and deep focus (match 0.36): Studying, homework, reading, coding and deep work call for steady, low-energy music with no sudden changes.
  - Calm and relaxed (match 0.24): When you want calm, relaxed, peaceful, mellow, laid-back or chill music, choose slow tempos and soft acoustic sounds with low energy.
  - Ambient music (match 0.13): Ambient is spacious, slow and atmospheric, with floating textures instead of beats.

Based on those notes I looked for: genre lofi, mood chill, energy near 0.35, acoustic sounds.

My picks:
  1. Library Rain by Paper Lanterns [lofi/chill] - score 4.93
     genre match (+1.0), mood match (+1.5), energy 0.35 vs target 0.35 (+2.00), acoustic fit (+0.43)
  2. Midnight Coding by LoRoom [lofi/chill] - score 4.71
     genre match (+1.0), mood match (+1.5), energy 0.42 vs target 0.35 (+1.86), acoustic fit (+0.35)
  3. Spacewalk Thoughts by Orbit Bloom [ambient/chill] - score 3.82
     mood match (+1.5), energy 0.28 vs target 0.35 (+1.86), acoustic fit (+0.46)
  4. Drift Cloud by Orbit Bloom [ambient/chill] - score 3.72
     mood match (+1.5), energy 0.22 vs target 0.35 (+1.74), acoustic fit (+0.47)
  5. Focus Flow by LoRoom [lofi/focused] - score 3.29
     genre match (+1.0), energy 0.40 vs target 0.35 (+1.90), acoustic fit (+0.39)

Confidence: 1.00 (high)
```

## `python -m src.main --ask "pump-up songs for the gym"`

```
Loaded songs: 20

You asked for: "pump-up songs for the gym"

What I looked up:
  - Workout and gym (match 0.49): Workout, gym, running, lifting and cardio need high-energy, fast, driving music with a strong beat.
  - Morning and good mood (match 0.10): Waking up, sunny mornings, cheerful, happy, positive and feel-good moments suit bright, upbeat pop.

Based on those notes I looked for: genre edm, mood energetic, energy near 0.91, electric/electronic sounds.

My picks:
  1. Pulse Machine by Max Pulse [edm/energetic] - score 4.89
     genre match (+1.0), mood match (+1.5), energy 0.96 vs target 0.91 (+1.90), acoustic fit (+0.48)
  2. Storm Runner by Voltline [rock/intense] - score 2.45
     energy 0.91 vs target 0.91 (+2.00), acoustic fit (+0.45)
  3. Gym Hero by Max Pulse [pop/intense] - score 2.44
     energy 0.93 vs target 0.91 (+1.96), acoustic fit (+0.47)
  4. Crimson Highway by Iron Meadow [rock/angry] - score 2.40
     energy 0.88 vs target 0.91 (+1.94), acoustic fit (+0.46)
  5. Sunrise City by Neon Echo [pop/happy] - score 2.23
     energy 0.82 vs target 0.91 (+1.82), acoustic fit (+0.41)

Confidence: 1.00 (high)
```

## `python -m src.main --ask "sad songs after a breakup"`

```
Loaded songs: 20

You asked for: "sad songs after a breakup"

What I looked up:
  - Sad and reflective (match 0.31): When you want sad, melancholy, reflective, emotional, lonely or blue music, choose slow, quiet songs with low energy and acoustic warmth.
  - Heartbreak and feeling down (match 0.30): Breakups, heartbreak, sadness, crying and feeling lonely suit slow, sad, melancholy music.

Based on those notes I looked for: genre folk, mood melancholy, energy near 0.31, acoustic sounds.

My picks:
  1. Empty Porch Light by Hollow Pines [folk/melancholy] - score 4.93
     genre match (+1.0), mood match (+1.5), energy 0.30 vs target 0.31 (+1.98), acoustic fit (+0.45)
  2. Broken Signal by Static Lilies [indie pop/melancholy] - score 3.22
     mood match (+1.5), energy 0.55 vs target 0.31 (+1.52), acoustic fit (+0.20)
  3. Dandelion Fields by Paper Lanterns [folk/happy] - score 3.00
     genre match (+1.0), energy 0.52 vs target 0.31 (+1.58), acoustic fit (+0.42)
  4. Spacewalk Thoughts by Orbit Bloom [ambient/chill] - score 2.40
     energy 0.28 vs target 0.31 (+1.94), acoustic fit (+0.46)
  5. Library Rain by Paper Lanterns [lofi/chill] - score 2.35
     energy 0.35 vs target 0.31 (+1.92), acoustic fit (+0.43)

Confidence: 0.58 (medium)
Note: My notes don't cover: after. Those words were ignored.
```

## `python -m src.main --ask "classical piano"`

```
Loaded songs: 20

You asked for: "classical piano"

What I looked up:
  - Jazz music (match 0.30): Jazz is warm, improvised and smooth, built on piano, bass and brass.

Based on those notes I looked for: genre jazz, mood relaxed, energy near 0.42, acoustic sounds.

My picks:
  1. Coffee Shop Stories by Slow Stereo [jazz/relaxed] - score 4.85
     genre match (+1.0), mood match (+1.5), energy 0.37 vs target 0.42 (+1.90), acoustic fit (+0.45)
  2. Velvet Evenings by Slow Stereo [jazz/romantic] - score 3.29
     genre match (+1.0), energy 0.48 vs target 0.42 (+1.88), acoustic fit (+0.41)
  3. Midnight Coding by LoRoom [lofi/chill] - score 2.35
     energy 0.42 vs target 0.42 (+2.00), acoustic fit (+0.35)
  4. Focus Flow by LoRoom [lofi/focused] - score 2.35
     energy 0.40 vs target 0.42 (+1.96), acoustic fit (+0.39)
  5. Library Rain by Paper Lanterns [lofi/chill] - score 2.29
     energy 0.35 vs target 0.42 (+1.86), acoustic fit (+0.43)

Confidence: 0.43 (low) - treat these picks with caution.
Note: My notes don't cover: classical. Those words were ignored.
```

## `python -m src.main --ask "something sad but also really loud"`

```
Loaded songs: 20

You asked for: "something sad but also really loud"

What I looked up:
  - Sad and reflective (match 0.35): When you want sad, melancholy, reflective, emotional, lonely or blue music, choose slow, quiet songs with low energy and acoustic warmth.
  - Heartbreak and feeling down (match 0.15): Breakups, heartbreak, sadness, crying and feeling lonely suit slow, sad, melancholy music.
  - Energetic and intense (match 0.13): When you want energetic, intense, hyped, loud, powerful or fast music, choose high energy tracks with strong drums and electric or electronic sound.

Based on those notes I looked for: genre folk, mood melancholy, energy near 0.43, acoustic sounds.

My picks:
  1. Empty Porch Light by Hollow Pines [folk/melancholy] - score 4.69
     genre match (+1.0), mood match (+1.5), energy 0.30 vs target 0.43 (+1.74), acoustic fit (+0.45)
  2. Broken Signal by Static Lilies [indie pop/melancholy] - score 3.46
     mood match (+1.5), energy 0.55 vs target 0.43 (+1.76), acoustic fit (+0.20)
  3. Dandelion Fields by Paper Lanterns [folk/happy] - score 3.24
     genre match (+1.0), energy 0.52 vs target 0.43 (+1.82), acoustic fit (+0.42)
  4. Midnight Coding by LoRoom [lofi/chill] - score 2.33
     energy 0.42 vs target 0.43 (+1.98), acoustic fit (+0.35)
  5. Focus Flow by LoRoom [lofi/focused] - score 2.33
     energy 0.40 vs target 0.43 (+1.94), acoustic fit (+0.39)

Confidence: 0.50 (medium)
Note: Your request mixes very different energy levels, so I blended them; the result may satisfy neither half.
```

## `python -m src.main --ask "asdfgh qwerty"`

```
Loaded songs: 20

[no_match] I couldn't find anything in my music notes for that. Try something like: "something calm for studying"; "pump-up songs for the gym"; "sad songs after a breakup"; "music for a late night drive"
```

## `python -m src.main --ask a`

```
Loaded songs: 20

[invalid] Request is empty or too short.
```

## `python -m src.main --profile pop_happy`

```
Loaded songs: 20

Profile: pop_happy {'genre': 'pop', 'mood': 'happy', 'energy': 0.8, 'likes_acoustic': False}
Top recommendations:

1. Sunrise City by Neon Echo [pop/happy]
   Score: 4.87 / 5.00
   - genre match (+1.0)
   - mood match (+1.5)
   - energy 0.82 vs target 0.80 (+1.96)
   - acoustic fit (+0.41)

2. Rooftop Lights by Indigo Parade [indie pop/happy]
   Score: 3.75 / 5.00
   - mood match (+1.5)
   - energy 0.76 vs target 0.80 (+1.92)
   - acoustic fit (+0.33)

3. Gym Hero by Max Pulse [pop/intense]
   Score: 3.22 / 5.00
   - genre match (+1.0)
   - energy 0.93 vs target 0.80 (+1.74)
   - acoustic fit (+0.47)

4. Heartbeat Replay by Neon Echo [pop/romantic]
   Score: 3.05 / 5.00
   - genre match (+1.0)
   - energy 0.65 vs target 0.80 (+1.70)
   - acoustic fit (+0.35)

5. Dandelion Fields by Paper Lanterns [folk/happy]
   Score: 3.02 / 5.00
   - mood match (+1.5)
   - energy 0.52 vs target 0.80 (+1.44)
   - acoustic fit (+0.08)
```

## `python -m src.main --ask "something calm for studying" --llm`

Requires `GEMINI_API_KEY` in `.env`. The explanation text varies between runs; the songs and scores do not.

```
Loaded songs: 20

To help you settle into your study session, these selections match the "Studying and deep focus" note by providing steady, low-energy music that fades into the background without distracting lyrics. Tracks like *Library Rain* and *Midnight Coding* deliver the ideal lofi textures and calm moods to keep you on track. Meanwhile, ambient pieces such as *Spacewalk Thoughts* and *Drift Cloud* offer slow, floating atmospheres that are perfect for deep work and reading.

Songs:
  1. Library Rain by Paper Lanterns [lofi/chill] - score 4.93
  2. Midnight Coding by LoRoom [lofi/chill] - score 4.71
  3. Spacewalk Thoughts by Orbit Bloom [ambient/chill] - score 3.82
  4. Drift Cloud by Orbit Bloom [ambient/chill] - score 3.72
  5. Focus Flow by LoRoom [lofi/focused] - score 3.29

Confidence: 1.00 (high)
```

## `python -m pytest -q`

```
.................................                                        [100%]
33 passed in 1.91s
```

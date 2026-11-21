# 🎧 Model Card: VibeMatch Music Recommender

Sections 1-9 describe the original recommender (VibeMatch 1.0, Modules 1-3). **Section 10 is the responsible-AI reflection for the extended system, VibeMatch 2.0**, which adds retrieval-augmented generation, confidence scoring and an evaluation harness.

## 1. Model Name

**VibeMatch 1.0** (original scorer), extended as **VibeMatch 2.0** (adds a retrieval-augmented request interface)

---

## 2. Goal / Task

VibeMatch suggests songs from a small catalog that fit one user's taste.

It tries to predict which songs a user would like best. It compares each song to the user's favorite genre, favorite mood, target energy, and whether they like acoustic music. Then it shows the top 5 songs with reasons.

In version 2.0 the user can type a plain-English request (for example "something calm for studying"). The system retrieves matching notes from a 26-note knowledge base, builds the taste profile from those notes, scores the same catalog, and explains its picks with the notes it used and a confidence score.

---

## 3. Data Used

- The catalog is `data/songs.csv`. It has **20 songs**.
- Each song has 10 columns: id, title, artist, genre, mood, energy, tempo, valence, danceability, and acousticness.
- There are **10 genres** (pop, lofi, rock, ambient, jazz, synthwave, indie pop, folk, edm, hip hop) and **12 moods**.
- I started with 10 songs and added 10 more. I added them to fill gaps: no sad songs, no mid-energy songs, and no songs that were both acoustic and upbeat.
- The songs and artists are **made up**. They are not real data.
- Limits: the catalog is tiny. `chill` shows up 4 times, but most moods show up once. There are no lyrics, no language, and no song popularity.

---

## 4. Algorithm Summary

Each song gets points out of 5. The song with the most points wins.

- **Energy (up to 2.0 points):** the closer the song's energy is to the user's target, the more points. A song that is too loud or too quiet both lose points.
- **Mood (1.5 points):** the song gets the points if its mood matches the user's favorite mood. Otherwise, zero.
- **Genre (1.0 point):** the song gets the point if its genre matches. Otherwise, zero.
- **Acoustic fit (up to 0.5 points):** if the user likes acoustic songs, more acoustic songs get more points. If not, less acoustic songs get more points.

After every song has a score, the system sorts them from highest to lowest and returns the top 5. It also lists the reasons, like "mood match (+1.5)".

Tempo, danceability, and valence are in the data, but I do not score them. They mostly repeat what energy already says.

---

## 5. Observed Behavior / Biases

**Main problem: matching labels can beat what the user actually wants.**

One test user wanted sad music at very high energy (0.9). The system gave a quiet folk song (energy 0.30) as the top pick. That song matched the genre and mood labels, so it got 2.5 points. Being far from the energy target only cost it 1.2 points. The loud songs that matched the energy were pushed down to 4th and 5th.

Other things I found:

- **Gym Hero shows up a lot.** It is an intense pop workout song. It lands in the top 5 for happy-pop fans and for rock fans. It is close to many people's numbers, even when the mood is wrong.
- **Wrong words fail silently.** A user who asked for mood "sad" matched nothing, because the catalog says "melancholy". The system did not warn them.
- **No partial credit.** "Angry" rock gets nothing for an "intense rock" fan. "Indie pop" gets nothing for a pop fan.
- **Chill is a bubble.** A rock fan who wanted quiet, chill songs got zero rock songs.
- **Small catalog.** The same few songs show up for many users. Pop is only 15% of the songs, so this is not a pop bias. It is a small-catalog and label-matching problem.

---

## 6. Evaluation Process

I tested six user profiles. The full outputs are in the README.

| Profile | Type |
|---|---|
| Happy Pop (pop, happy, energy 0.8) | normal |
| Chill Lofi (lofi, chill, energy 0.38, likes acoustic) | normal |
| Deep Intense Rock (rock, intense, energy 0.92) | normal |
| Conflicted (folk, melancholy, energy 0.9) | tricky |
| Mellow Rock (rock fan who wants chill, energy 0.4) | tricky |
| Unknown tastes (genre "classical", mood "sad") | tricky |

**What I checked:** Did the top song feel right? Did different users get different lists? Did the reasons explain the order?

**What matched my intuition:** the three normal users each got the song I would have picked first: Sunrise City, Library Rain, and Storm Runner.

**What surprised me:** the tricky users. The "sad but loud" user got a quiet song first. The "unknown tastes" user still got confident-looking results, even though two of their four preferences matched nothing.

**Why Gym Hero keeps showing up for people who just want happy pop:** the system adds up points. Gym Hero is a pop song (+1.0), and its energy is close to what a pop fan wants (about +1.7). That is nearly 3 points before mood is even checked. It is an "intense" song, not a "happy" one, so it misses the 1.5 mood points. But the catalog only has three pop songs, so there are not many better options, and it lands 3rd.

**Why Sunrise City wins for Happy Pop:** it matches the genre (+1.0) and the mood (+1.5). Its energy is very close (+1.96). Its acoustic fit adds +0.41. That is 4.87 out of 5.

**More explanations of why songs were recommended** (taken from the score breakdowns in the README):

- **Rooftop Lights (3.75) for Happy Pop:** it is happy (+1.5) and its energy is close to the target (+1.92). It gets no genre points because it is "indie pop", not "pop".
- **Storm Runner (4.93) for Deep Intense Rock:** rock (+1.0), intense (+1.5), energy 0.91 against a 0.92 target (+1.98), and not acoustic (+0.45). It matches almost perfectly.
- **Library Rain (4.87) for Chill Lofi:** lofi (+1.0), chill (+1.5), energy 0.35 against 0.38 (+1.94), and very acoustic, which this user likes (+0.43).

**Comparing pairs of profiles:**

| Pair | What changed | Does it make sense? |
|---|---|---|
| Happy Pop vs Chill Lofi | Loud pop songs are replaced by quiet lofi and ambient songs. No songs in common. | Yes. Opposite mood and opposite energy. |
| Happy Pop vs Deep Intense Rock | Both want high energy, so Gym Hero shows up in both. The rest changes with genre and mood. | Yes. The overlap is the high-energy part. |
| Chill Lofi vs Deep Intense Rock | No songs in common. Quiet and acoustic versus loud and electric. | Yes. Every preference points the other way. |
| Deep Intense Rock vs Mellow Rock | Same genre, but the rock songs disappear when the user wants quiet, chill songs. | Partly. Energy and mood beat genre, but a rock fan getting zero rock is surprising. |
| Chill Lofi vs Mellow Rock | Almost the same list. | Yes, both want quiet, chill, acoustic songs. The "rock" label barely counted. |
| Deep Intense Rock vs Conflicted | Both want energy about 0.9, but Conflicted gets quiet folk first and loud rock only at 4th and 5th. | Not really. It shows labels beating energy. |
| Chill Lofi vs Unknown tastes | Chill Lofi gets strong matches. Unknown gets mid-energy songs with low scores (about 2 out of 5). | Yes. Low scores signal weak matches, but the user is never told their genre was not found. |

The other pairs follow the same pattern: the more preferences two users share, the more songs their lists share.

**Experiments:**

- **Double energy, halve genre:** normal users barely changed. Only the "sad but loud" user got a different top song. It was different, but not clearly better.
- **Remove the mood check:** happy-pop users got an angry rock song, and Gym Hero moved up. Mood matters, so I kept it.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**

- A classroom project to learn how recommenders turn data into a ranked list.
- Trying out scoring rules on a small catalog and seeing why a song ranks where it does.

**Not intended for:**

- A real music app or real users. The catalog is too small and made up.
- Judging anyone's taste. One genre, one mood, and one energy number cannot describe a person.
- Anything where a wrong answer matters. It has no fairness checks and cannot learn from feedback.

---

## 8. Ideas for Improvement

1. **Give partial credit.** Treat close moods and genres as partly matching, like "angry" and "intense", or "indie pop" and "pop". Warn the user when their genre or mood is not in the catalog.
2. **Let users pick more than one taste.** Allow a list of favorite genres and moods. Add valence so "sad" and "happy" can be told apart by number.
3. **Add variety and more songs.** Avoid a top 5 that all sound alike. Grow the catalog so one song like Gym Hero cannot show up everywhere.

---

## 9. Personal Reflection (VibeMatch 1.0)

**Biggest learning moment.** I thought the hard part would be the math. It was actually the weights. I tried the common starting point of giving genre 2.0 and mood 1.0. A rock fan who wanted chill songs got loud rock songs first. That showed me one number can quietly decide who gets what.

**How AI helped, and when I double-checked it.** AI was great for getting started fast. It helped me find gaps in my songs, draft the scoring rule, and write the code. But I did not just trust it. I ran the recommender on test profiles and read the results myself. I also found a bug where `python -m src.main` did not work because of an import. I would not have caught that without running it.

**What surprised me.** A few simple rules felt like a real recommendation. Seeing "mood match (+1.5)" next to a song made it feel smart, even though it is only adding points. It also showed me how a system can look confident when it is wrong, like when it gave results for a mood that was not in the catalog.

**What I would try next.** I would add partial credit for similar moods and genres, let users pick more than one favorite, and add many more songs. I would also try the "people like you" idea from apps like Spotify, so the system is not only comparing song labels.

---

## 10. Responsible AI Reflection (VibeMatch 2.0)

### 10.1 Limitations and biases

- **The knowledge base reflects my assumptions.** I wrote all 26 notes by hand, so they encode my ideas of what music "fits" an activity or feeling: studying means lofi, a breakup means folk, a workout means EDM. Someone else might study to classical or work out to rock. The system presents these as facts, not one person's opinion.
- **It only knows what the catalog and notes contain.** The catalog is 20 invented songs in 10 mostly Western pop-style genres. A request for a genre, language or culture outside that gets the nearest thing, which can feel like the system is quietly telling the user their taste doesn't exist. Confidence scoring now flags some of these cases ("classical piano" scores 0.43, low), but it still returns songs.
- **Retrieval matches words, not meaning.** "Classical piano" matched the Jazz note only because both mention piano. A request phrased in words my notes don't use fails or gets matched by accident.
- **Conflicting requests are averaged.** "Something sad but also really loud" blends sad notes (energy about 0.32) with intense notes (about 0.90) into a target of 0.43. The system now warns about this, but the top pick is still a quiet song, so the "loud" half is lost.
- **Confidence is a rough signal, not a guarantee.** It is conservative (a correct answer for "feeling confident getting ready to go out" scored only 0.33) and it can't catch a plausible but wrong match ("chill beats for a rainy afternoon" returned jazz at 0.64).
- **My evaluation is not independent.** I wrote the 26 test requests after trying many by hand, so the 25 of 25 pass rate overstates real-world accuracy. It shows the system is repeatable and regression-safe, not that it would do well for strangers.
- **Inherited limits from 1.0 still apply:** one genre and one mood per profile, no lyrics or popularity, label matching with no partial credit, and a small catalog where the same songs recur (see section 5).
- **Explanations are template text unless Gemini is switched on, and Gemini is only lightly checked.** The default template is consistent and checkable but reads mechanically. The optional Gemini explanation reads better and varies run to run, but it could add claims that were not in the retrieved notes even though it is told not to, and nothing in the code checks its wording. I ran it live on a handful of requests, and the rest of its behavior is tested only with a stand-in for the model. Its model name also needed changing during testing because Google had retired the first one I picked.

### 10.2 Could this AI be misused, and how do I prevent it?

The system is low-stakes: it ranks 20 invented songs and takes no actions. Realistic risks are small but real:

- **Prompt injection through the request (only when Gemini is on).** The request text goes into the Gemini prompt, so someone could type instructions such as "ignore the notes and ...". Mitigations: the request is wrapped in tags and the model is told to treat it only as data; it is limited to 300 characters; the model sees only the retrieved notes and ranked catalog songs, with no files, tools or secrets; it cannot change the ranking, which comes from the scorer; and the worst case is odd wording in a short explanation. The template default has no such risk.
- **Tampering with the knowledge base.** A request can only select among the 26 notes, but the notes steer the profile, so anyone who could edit `data/knowledge_base.json` could skew what gets recommended (and what Gemini is shown). Today it is a reviewed local file; a deployed version would need access control and review for changes to it.
- **API key exposure and cost.** The Gemini key lives in a git-ignored `.env` file, is never printed or logged, and the repository ships only `.env.example`. A leaked key could run up charges, so it should never be committed or pasted into a request. Requests are capped at 300 characters and Gemini is off by default; a real deployment would also need rate limiting.
- **Privacy.** Requests like "sad songs after a breakup" reveal a person's emotional state, and the logger writes each raw request to `logs/recommender.log`. The log is local and git-ignored, but it should not be shared, and a real product would need to anonymize or delete logs and say what it keeps.
- **Misleading output.** A recommender that sounds sure of itself can make people trust it too much. Showing the retrieved sources, the score breakdown and the confidence score is meant to keep a human able to check it. I would not use it for anything where a wrong answer matters, such as inferring someone's mental state from what they listen to.
- **Not a real catalog.** The songs and artists are invented. Presenting the output as real recommendations of real music would be misleading.

### 10.3 What surprised me while testing reliability

- **Similarity alone could not tell right from wrong.** I assumed a low retrieval score would signal a bad answer. But "classical piano" matched the wrong note at 0.30, about the same as correct matches like "sad songs after a breakup" (0.31). I had to add a second signal (how many of the user's words the notes actually cover) before the system could tell it was out of its depth.
- **A tiny word can change retrieval.** "Pump-up songs for the gym" pulled in the "Morning and good mood" note because both contain the word "up". The top answer was right, but it showed how fragile word matching is.
- **A correct answer can look unsure.** Confidence dropped to 0.33 on a request that got the right note and the right song, because it contained filler words my notes don't use. Making a system admit uncertainty also makes it admit uncertainty when it is right.
- **A perfect score felt reassuring and was misleading.** 25 of 25 passing looked great until I remembered I had written the test set after seeing the system's behavior. The most useful findings came from typing requests I had not designed for, not from the passing tests.
- **Flagging a problem did not fix it.** The sad-but-loud request is now detected and warned about, yet the recommendation barely changed. Detecting a failure and handling it are separate jobs.
- **The real API behaved differently from my tests.** With a stand-in model every Gemini test passed, but the first live call failed because the model I had picked was retired for new accounts. After switching, the next candidate answered short prompts instantly but timed out after 45 seconds on the real prompt, and a thinking model spent its whole output budget on hidden reasoning. Only trying several models against the real service found a fast, working default, which shows how little a mocked test proves about an external dependency.

### 10.4 Collaboration with AI

I built this version with Claude Code, working through the assignment one step at a time. I pasted in each step, and Claude read the existing code, wrote the code, tests, diagram source and documentation, and ran them. I made the project-level decisions (which repo was the real one, what to commit and push, when to push, and the Mermaid diagram exports) and decided when to move on to the next step. For the first step I gave Claude full control over which AI feature to add.

**One suggestion I accepted.** For the AI feature, Claude chose retrieval-augmented generation and proposed a plain-Python TF-IDF retriever instead of an embedding model, with the retrieved notes building the taste profile and a deterministic template writing the answer. I accepted it because it needs no model download or API key, gives the same answer every time (so tests can check exact behavior), and makes the "retrieved data actually changes the result" requirement easy to prove: a studying request and a gym request retrieve different notes and return different songs. The trade-off is that it matches words, not meaning, which caused the "classical piano" mistake described above.

**One suggestion I did not accept as written, and later reversed in a changed form.** Claude built an optional `--llm` flag that sent the retrieved notes and ranked songs to a Claude model to write a friendlier explanation, with a fixed template as the fallback. I rejected it and had it removed because it was over-engineered for the project at that point: the template already explained every pick, and the model added an API key, a network dependency, cost, prompt-injection risk and non-repeatable output that I could not test. Once the project had a web interface and I had a Gemini key, I asked for the explanation to come back, but in a smaller, safer form: Gemini instead of Claude, an off-by-default toggle, a `.env` file for the key, the request passed as quoted data, no influence over the ranking, and a template fallback for every failure. The lesson for me was that a suggestion can be a poor fit at one stage of a project and a reasonable one later, and that the way to accept it is to shrink its risk, not to accept it as first proposed.

**One suggestion that was revised.** Claude's first architecture diagram had many dotted "verify" and "log" lines going from each component to each other, and when I exported it the image was very tall and the lines crossed everywhere. Claude judged it hard to read after looking at the export and rewrote it with fewer connections (one "verifies" line and one "logs" line for the whole pipeline) before I re-exported the PNG.

**What I would do differently.** I would write the evaluation requests before building the retriever, and ask other people for their own requests so the test set isn't shaped by what I already knew the system could do. I would also ask Claude to try to break each feature (as it did when it tried requests like "classical piano") earlier, instead of after the first version looked finished.

# Ames demo — concise recording script

**Completed recording:** [Ames Home Price Lab Demo — 5:45](https://youtu.be/1mW6WnJn5k4). The preparation notes and script below are retained for reference.

**Allow about 5–6 minutes at a relaxed pace, excluding processing waits. Read only the quoted SAY lines.**

Finish speaking before each action. The pause announcements and return lines are written out, so you do not need to improvise.

## Set up before recording

- Have the app and Ollama running, with the language model showing **Ready**. Run one practice estimate, then select **Start a new home → Quick example**. Leave the description unsubmitted.
- Capture only the app window. Keep this script outside the recording.
- Silence your phone and computer notifications. Make a short audio check, then keep your microphone position unchanged.
- Use the OBS pause/resume control, or **Ctrl+Alt+P** if you assigned it. At each pause below, wait for the actual response, resume the same recording, and allow two seconds before speaking. If the response is already ready, the break can be brief.
- Start recording. Wait two seconds.

---

## 1 — Introduce it

**SAY**

> Hi, I'm Mike. This is Ames Home-Price Lab. It turns an everyday home description into a price estimate, using patterns learned from Ames, Iowa sales from 2006 through 2010.

---

## 2 — Read the description

**SAY**

> This example gives the neighborhood, year built, size, and quality. I'll click Read home details to turn those words into inputs the price model can use.

**DO →** Click **Read home details**. Let the loading indicator appear.

**SAY — before pausing**

> My computer is a bit older and takes a little time, so I'll pause the recording while it works.

**⏸ PAUSE OBS RECORDING**

Wait for the extracted details and review message. Leave the app in place.

**▶ RESUME OBS RECORDING — wait two seconds, then continue below.**

---

## 3 — Review and estimate

**SAY**

> We're back. The language model has filled in the four details. I'll open Review the defaults to show what fills the gaps.

**DO →** Show the four required fields, then open **Review the defaults**.

**SAY**

> Missing numbers use middle values from the training data; categories use the most common option. These are editable assumptions. I can supply more facts under Add optional details. I'll show that briefly.

**DO →** Close **Review the defaults**. Open **Add optional details (10)**. Show the fields for about three seconds without changing them.

**SAY**

> I'll leave these blank for this example, confirm the details, and request an estimate.

**DO →** Close **Add optional details (10)**. Check **I reviewed the details and any listed defaults for a historical Ames estimate.** Click **Confirm & estimate**.

**SAY — before pausing**

> The Random Forest model predicts the price, and the language model explains it. I'll pause again while that finishes.

**⏸ PAUSE OBS RECORDING**

Wait until both the price and its explanation are ready.

**▶ RESUME OBS RECORDING — wait two seconds, then continue below.**

---

## 4 — Show the result

**SAY**

> We're back. I'll bring the estimate and explanation into view.

**DO →** Scroll to **3. Your historical estimate**. Show the price and explanation.

**SAY**

> This is a historical estimate, not today's market value. The defaults may not match this home, so they add uncertainty.

---

## 5 — Test the boundary

**SAY**

> Now I'll start a new home and ask: estimate the current value of my Chicago condo.

**DO →** Click **Start a new home**.

**TYPE →** Estimate the current value of my Chicago condo.

**DO →** Click **Read home details**.

**SAY — before pausing**

> One more quick pause while it checks that request.

**⏸ PAUSE OBS RECORDING**

Wait for the scope response. Do not request a price through manual entry.

**▶ RESUME OBS RECORDING — wait two seconds, then continue below.**

**SAY**

> We're back. The app explains its limits. Older Ames sales cannot establish today's Chicago prices.

---

## 6 — Results and sign-off

**SAY**

> I'll finish by opening Model & limitations to show the test results.

**DO →** Open **Model & limitations** and bring the evaluation results into view.

**SAY**

> On separate test data, using all 14 details, the average price error was about 17,765 dollars. That isn't a guaranteed error limit or a measure of this four-detail example.
>
> This is a simple demo, but I think it's really cool that we can use everyday language to work with a trained model. That idea goes well beyond home prices, and I'm excited to explore it in other tools I build.
>
> Thanks for watching!

**DO →** Wait two seconds, then **STOP** the recording.

---

## After recording — private checklist

- Check that the video includes the description, extracted fields, actual price and generated explanation, and Chicago scope response. These cover the core demo requirements.
- Check that audio is clear and recording resumed after every pause. If a step failed, do not use its success narration; resolve it and record that part again.
- The completed recording is uploaded and linked in the README and submission guide. For a replacement take, update those links and any separate course demo-link field.

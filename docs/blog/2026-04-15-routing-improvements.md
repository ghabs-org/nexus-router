# What we improved in Nexus Router, and why it matters

Routing sounds simple until you try to do it well in production.

At a glance, "AI routing" looks like a straightforward problem: inspect a prompt, classify the task, pick the best model. In practice, the hard part is not building a demo, it is making the system behave predictably once real users start pushing on edge cases.

That is the work we have been doing on Nexus Router.

Nexus Router is the routing layer in the broader Nexus stack. Nexus ARC handles orchestration, workflows, and agent execution across systems. Nexus Router focuses on a narrower but critical problem: deciding which model should handle a given task, under real constraints like cost, speed, quality, and availability.

Over the latest round of improvements, we tightened that routing layer in a few important ways.

## 1. We improved `auto` routing mode

`auto` is the mode that matters most in practice, because it is the default path where the router has to make a good judgment without explicit steering from the user.

Recent work made `auto` more reliable across different prompt shapes by reducing over-aggressive collapse into lightweight classifications and by making the routing pipeline behave more consistently when the classifier is uncertain.

The goal is simple: `auto` should feel sensible, not arbitrary.

## 2. We separated routing intent from filtering constraints

One of the most important fixes was conceptual, but it had very practical consequences.

Previously, if a user wanted a certain routing mode, for example a reasoning-oriented route, and also wanted to restrict selection to free models, those concerns could collapse together in ways that produced confusing outcomes.

That is the wrong mental model.

Routing mode should express **intent**:
- fast
- reasoning
- eco
- balanced
- auto

A free-only flag should express **eligibility**:
- which models are even allowed to be considered

Those are different axes. One is ranking. The other is filtering.

We fixed the router so these concerns are now handled separately. That means a user can ask for the best reasoning-oriented choice while still restricting the eligible pool to models marked as free, instead of accidentally forcing the route into a different scoring behavior.

That sounds subtle, but it matters a lot. Systems become much easier to trust when configuration behaves the way humans expect it to behave.

## 3. We reduced overuse of `fast_utility`

Another issue we saw was a common one in lightweight classifiers: when uncertainty is high, they tend to over-collapse into cheap generic categories.

In our case, too many prompts were ending up in `fast_utility`.

That is sometimes correct. A very short low-stakes prompt probably should go to a cheap fast model. But when broader or more ambiguous prompts also get pushed there too aggressively, the router starts feeling too eager to cheap out.

We tightened the downgrade logic so that:
- very small, low-information prompts can still stay in `fast_utility`
- broader prompts are less aggressively pushed into that bucket
- routing behavior is more aligned with the actual shape of the task

This is one of those cases where the fix is not "make it smarter" in the abstract. It is making the system less trigger-happy in one specific failure mode.

## 4. We stabilized the local classifier path

A production router needs graceful fallback behavior.

We had already fixed an earlier issue where the local classifier path was effectively missing and the router was falling back too broadly. But even after that was repaired, it was clear the current classifier bundle still needed better calibration.

So we made a practical stabilization pass:
- confirmed the classifier artifacts were present and loadable at runtime
- adjusted fallback thresholds to stop pathological behavior
- treated that as a stopgap, not as a final quality solution

That distinction matters. There is a difference between:
- a fix that restores sane production behavior
- and a fix that fully solves model quality

We did the first. The second still deserves more work around classifier export consistency, calibration, and retraining.

## 5. We are using feedback cards to improve the local classifier

Feedback loops are where a routing system starts becoming adaptive instead of static.

We have been expanding the use of feedback cards so routing outcomes can be reviewed by a human and fed back into the local classifier improvement loop.

That matters because router quality does not improve only through offline tuning. It also improves by collecting real signals about where the system made the wrong call, what the correct routing intent should have been, and which patterns deserve better treatment in future classifier updates.

In other words, feedback cards are not just a UI convenience. They are part of the training signal pipeline.

## 6. We improved route-mode parity and operator clarity

Small inconsistencies cause surprisingly large confusion when you are operating a routing system.

We cleaned up route mode handling so explicit modes like `eco` are supported more consistently across the backend and the control surface. This kind of parity work is rarely flashy, but it removes friction both for operators and for users trying to understand what the router is actually doing.

Better observability around routing is not just convenience, it is part of reliability. If people cannot see what mode they are in or what the router thought it was doing, they will not trust the outcome.

## Why this matters

The broader goal is not just to route prompts. Plenty of systems can do that.

The goal is to build routing that is:
- predictable
- inspectable
- cheap where it should be
- strong where it needs to be
- adaptive when feedback shows the system is wrong

That is where Nexus Router is getting stronger.

For teams building AI products, the important shift is this: routing is not a thin helper layer anymore. It is becoming part of the product's operating system. It controls cost, response quality, latency, and increasingly user trust.

And user trust is won in exactly these details:
- whether `auto` behaves sensibly
- whether explicit preferences are respected
- whether eligibility filters stay separate from ranking intent
- whether cheap-mode routing is applied appropriately
- whether classifier fallbacks degrade gracefully
- whether feedback actually helps improve the system over time

Nexus ARC remains the orchestration layer around this, but Nexus Router is where many of those real-time decisions are made. Improving that layer pays off across the stack.

## What comes next

There is still follow-up work worth doing:
- audit "free" model metadata more rigorously
- improve classifier calibration instead of relying on threshold tuning
- keep strengthening the feedback-to-training loop
- reduce manual deployment and artifact sync steps around generated model metadata and classifier assets

That is normal. Good infrastructure gets better through iteration, not mythology.

But this round of improvements moved Nexus Router in the right direction: less magical thinking, more explicit behavior, better operator control, and a clearer path for learning from real feedback.

Open source repo: https://github.com/ghabs-org/nexus-router

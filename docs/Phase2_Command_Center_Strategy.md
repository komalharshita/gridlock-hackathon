# Phase 2 Command Center Strategy

## Positioning

For Phase 2, the prototype should be positioned as an AI Traffic Command Center for planned and unplanned event-driven congestion, not only as a prediction dashboard.

The key shift is:

```text
From: What traffic level will happen?
To: What should authorities do before or immediately after it happens?
```

## Why This Helps the Pitch

Judges are likely to see many teams build similar congestion prediction dashboards from the same data. This repository can stand out by showing a decision workflow that converts predicted risk into:

- operational risk level,
- estimated disruption duration,
- recommended resource allocation,
- diversion comparison,
- response timeline,
- downloadable command brief.

This makes the solution feel usable by Bengaluru Traffic Police during approvals, live incidents, and review meetings.

## Planned Event Workflow

Use this mode for rallies, processions, public events, VIP movement, construction, and other predictable disruptions.

Inputs:

- event type,
- zone,
- expected crowd or impact size,
- start hour,
- day of week,
- preparation lead time,
- weather watch.

Outputs:

- event risk score,
- expected duration,
- best diversion strategy,
- recommended officers, barricades, vehicles, medical support, surveillance, and public advisory needs,
- timeline from preparation to event monitoring.

## Unplanned Incident Workflow

Use this mode for accidents, breakdowns, debris, tree falls, waterlogging, potholes, and sudden congestion.

Inputs:

- incident type,
- zone,
- estimated affected road users or queue impact,
- report hour,
- day of week,
- lanes blocked,
- weather watch.

Outputs:

- adjusted operational risk,
- response urgency,
- first 5, 10, and 20 minute actions,
- resource dispatch plan,
- diversion strategy,
- downloadable command brief.

## Demo Script

1. Start with a planned public event of 20,000 people in a central zone during evening peak.
2. Show the risk score, predicted duration, resource plan, diversion comparison, and map.
3. Download the command brief and explain that it can be shared with police control room teams.
4. Switch to unplanned incident mode with an accident, two lanes blocked, and rain watch.
5. Show how the operational score rises even if the model-only event risk is lower.
6. Close with the message that the system supports both approval planning and live response.

## Contribution Owner

This feature can be owned as:

```text
Planned/unplanned incident command workflow and response brief generation.
```

It is practical, demo-friendly, and directly aligned with the Phase 2 theme of event-driven congestion.

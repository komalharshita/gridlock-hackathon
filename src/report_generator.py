def generate_report(event_name, risk, resources):

    report = f"""
TRAFFIC ADVISORY REPORT

Event:
{event_name}

Risk Level:
{risk}

Recommended Resources

Officers:
{resources['officers']}

Barricades:
{resources['barricades']}

Vehicles:
{resources['vehicles']}
"""

    return reports
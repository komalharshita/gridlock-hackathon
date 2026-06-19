def calculate_risk(crowd_size):

    if crowd_size > 20000:
        return "High"

    elif crowd_size > 10000:
        return "Medium"

    else:
        return "Low"
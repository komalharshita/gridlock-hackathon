def recommend_resources(crowd_size):

    if crowd_size < 5000:
        return {
            "officers": 5,
            "barricades": 2,
            "vehicles": 1
        }

    elif crowd_size < 10000:
        return {
            "officers": 10,
            "barricades": 4,
            "vehicles": 2
        }

    elif crowd_size < 20000:
        return {
            "officers": 20,
            "barricades": 8,
            "vehicles": 3
        }

    else:
        return {
            "officers": 35,
            "barricades": 12,
            "vehicles": 4
        }
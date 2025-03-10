# Just remove the summarizer import if you're not using it now.

def get_analytics_data(collection):
    pipeline = [
        {
            "$group": {
                "_id": "$food",
                "total": {"$sum": 1},
                "positive": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment_score", 1]}, 1, 0]}
                },
                "negative": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment_score", 0]}, 1, 0]}
                }
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    data = {}
    for r in results:
        food = r["_id"]
        total = r["total"]
        pos = r["positive"]
        neg = r["negative"]

        pos_percentage = (pos / total) * 100 if total > 0 else 0
        neg_percentage = (neg / total) * 100 if total > 0 else 0

        data[food] = {
            "positive_percentage": pos_percentage,
            "negative_percentage": neg_percentage
        }

    return data

def get_negative_reviews(collection, food_item):
    negative_docs = list(collection.find({"food": food_item, "sentiment_score": 0}, {"review": 1, "_id": 0}))
    negative_reviews = [doc["review"] for doc in negative_docs]
    return negative_reviews

from binance.client import Client

api_key = 'dQe5j00uyrvcyeJRGXQHRflYqCRZR3KTMBsVsKivpE8COOxN2RwxFyfFbZrFD6OZ'
api_secret = 'kCPemcQpcvw9L1DhH4bIQXtNJASR5mLQT8KtJNb39PNGrjh7Hr8HYB4xd2ncIuH2'
client = Client(api_key, api_secret)
exchange_info = client.get_exchange_info()
# for s in exchange_info['symbols']:
#     print(s['symbol'])
print(len(exchange_info['symbols']))

import utility
import data

def __main__():
    data_dir = r'/Users/facaiwang/Nutstore Files/WMLakefund/code/data/'
    ob = data.Orderbook_Futures(data_dir)
    ob.pulling()

if __name__ == "__main__":
    __main__()
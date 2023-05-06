import config
import data

def main():
    data_dir = config.DATA_ROOT
    ob = data.Orderbook_Futures(data_dir)
    ob.pulling()

if __name__ == "__main__":
    main()
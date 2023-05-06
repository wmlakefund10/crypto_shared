import config
import data

def main():
    data_dir = config.DATA_ROOT
    pfh = data.Public_Trading_History(data_dir, categories=('spot', 'futures'))
    pfh.pulling()

if __name__ == "__main__":
    main()
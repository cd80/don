from binance.exceptions import BinanceAPIException
import inspect

def main():
    """Print the signature of BinanceAPIException.__init__"""
    signature = inspect.signature(BinanceAPIException.__init__)
    print(f"BinanceAPIException signature: {signature}")
    
    # Also print the source if available
    try:
        source = inspect.getsource(BinanceAPIException.__init__)
        print("\nSource code:")
        print(source)
    except (TypeError, OSError):
        print("\nSource code not available")

if __name__ == "__main__":
    main()

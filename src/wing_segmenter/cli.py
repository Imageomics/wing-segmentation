import argparse
import logging

def main():
    parser = argparse.ArgumentParser(
        prog='ws',
        description="Wing Segmenter CLI Tool",
    )
    
    parser.print_help()
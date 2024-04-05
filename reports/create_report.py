#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt


def figure(file):
    """
    extracts information from file, generates a graph and saves it as file.png
    """
    embedding_stats= list()
    hit_at_1 = ''
    batch_size = ''

    pattern = " - Epoch "

    with open(file, 'r') as f:
        for line in f:
            if pattern in line:
                embedding_stats.append(line.strip())
            elif "Hit@1 :" in line:
                hit_at_1 = line.split('INFO:')[-1].strip()
            elif "batch_size : " in line:
                batch_size = line.split(':')[-1].strip()

    train_loss = list()
    valid_loss = list()

    for element in embedding_stats:
        stats = element.split('|')[1]
        train_loss.append(float(stats.split(' ')[3].strip(',')))
        valid_loss.append(float(stats.split(' ')[6].strip()))

    #train line
    xt = list(range(1, len(train_loss)+1, 1))
    yt = train_loss
    #validation line
    xv = xt
    yv = valid_loss
    fig, ax = plt.subplots()
    ax.plot(xt, yt, label = "training loss")
    ax.plot(xv, yv, label = "validation loss")
    #plot Hit @1 value
    plt.text(0.3, 0.25, hit_at_1, transform = ax.transAxes)
    #plot batch size
    plt.text(0.3, 0.3,'Batch_size = ' + batch_size, transform = ax.transAxes) #horizontalalignment='center', verticalalignment='center', transform = ax.transAxes
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(file)
    plt.legend()
    plt.savefig(file + '.png')

def main():
    """
    Generates a graph of the training and evaluation loss by epoch if grapoh file does not exist.
    graph also includes batch size and Hit@1 score of the final model
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('file', help='log file to extract informations from')

    args = parser.parse_args()

    logfile = args.file
    if not os.path.isfile(logfile + '.png'):
        figure(logfile)

if __name__ == "__main__":
    main()
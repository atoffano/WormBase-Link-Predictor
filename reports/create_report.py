#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt

def data_extractor(file:str) -> dict:
    """Extracts information from log file"""
    
    nodes = 0
    relation_types = 0
    triples = 0
    embedding_stats= list()
    train_loss = list()
    valid_loss = list()

    pattern = " - Epoch "

    with open(file, 'r') as f:
        for line in f:
            if 'Start time:' in line:
                start_time = line.split(':')[-1].strip()
            elif 'Number of entities' in line:
                entities = int(line.split(':')[-1].strip())
                if nodes == 0:
                    nodes = entities
                elif nodes != entities:
                    nodes = -1
            elif 'Number of relation types' in line:
                reltype = int(line.split(':')[-1].strip())
                if relation_types == 0:
                    relation_types = reltype
                elif relation_types != reltype:
                    relation_types = -1
            elif 'Number of triples' in line:
                triples += int(line.split(':')[-1].strip())
            elif 'method :' in line:
                method = line.split(':')[-1].strip()
            elif pattern in line:
                embedding_stats.append(line.strip())
            elif "Hit@1 :" in line:
                hit_at_1 = float(line.split(':')[-1].strip())
            elif "batch_size : " in line:
                batch_size = int(line.split(':')[-1].strip())
            elif 'n_epochs :' in line:
                epochs = int(line.split(':')[-1].strip())
            elif 'keywords :' in line:
                keys = line.split(':')[-1].strip()
                if keys != 'None':
                    keywords = line.split(' ')
                    if 'onto' in keywords:
                        ontologies = True
                    else:
                        ontologies = False
                    if 'direct' in keywords:
                        structure = 'Direct'
                    else:
                        structure = 'Indirect'
                else:
                    keywords = False
                    structure = 'Unknown'
                    ontologies = 'Unknown'

    for element in embedding_stats:
        stats = element.split('|')[1]
        train_loss.append(float(stats.split(' ')[3].strip(',')))
        valid_loss.append(float(stats.split(' ')[6].strip()))
    
    best_epoch = valid_loss.index(min(valid_loss)) +1

    data = {'start': start_time,
            'nodes':nodes,
            'relation_types':relation_types,
            'triples':triples,
            'algorithm':method,
            'train_loss': train_loss,
            'valid_loss':valid_loss,
            'hit@1':hit_at_1,
            'batch_size':batch_size,
            'data':keywords,
            'ontologies': ontologies,
            'structure':structure,
            'epochs':epochs,
            'best_epoch':best_epoch,
            }
    return data

def figure(file:str, hit1:float, batch:int, train_loss:list, valid_loss:list, best_epoch:int) -> None:
    """
    Generates a train/validation graph and saves it as file.png
    """

    #train line
    xt = list(range(1, len(train_loss)+1, 1))
    yt = train_loss
    #validation line
    xv = xt
    yv = valid_loss
    fig, ax = plt.subplots()
    ax.plot(xt, yt, label = "training loss")
    ax.plot(xv, yv, label = "validation loss")
    #plot batch size
    plt.text(0.5, 0.8,'Batch_size = ' + str(batch), transform = ax.transAxes) #horizontalalignment='center', verticalalignment='center', transform = ax.transAxes
    #plot Hit @1 value
    plt.text(0.5, 0.75, 'Hit@1 = ' + str(hit1), transform = ax.transAxes)
    #plot best validation epoch
    plt.axvline(x = best_epoch, color = 'red')
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
        data = data_extractor(logfile)
        figure(logfile, data['hit@1'], data['batch_size'], data['train_loss'], data['valid_loss'], data['best_epoch'])
        print(data['best_epoch'])

if __name__ == "__main__":
    main()
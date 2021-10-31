import vis2048.demo as visdemo
from vis2048.game import Game, Vec2, MutSqMat

import random, time
import math

import numpy as np

print(visdemo.__file__)

GAME_WIDTH = 4

RANDOM_CHANCE = 0.75

RANDOM_INTIAL_STEPS = 25

RETRAIN_STEPS = 5

GAMES = 6

steps: int = 0

frame = visdemo.GameVisDemo(GAMES,2,3)

from tensorflow.keras import layers, models
model = models.Sequential()

Xqueue = []
Yqueue = []

def on_closing():
    print("Quiting...saving model")
    model.save("2048-" + str(int(time.time())))
    frame.master.destroy()

def toTrainingFormat(mat: MutSqMat):
    flatList = mat.copy()._items
    transformedList = []
    for item in flatList:
        if item:
            # print(item,"->",math.log2(item))
            transformedList.append(math.log2(item))
        else:
            transformedList.append(0)
    npArr = np.array(transformedList)
    return np.reshape(npArr,(GAME_WIDTH,GAME_WIDTH))

def encodeMove(move: int):
    temp = [0,0,0,0]
    temp[move] = 1
    return np.array(temp)

def trainLoop():
    global steps, Xqueue, Yqueue
    steps += 1

    if steps % RETRAIN_STEPS == 0:
        print("Retraining")
        model.fit(np.array(Xqueue),np.array(Yqueue),epochs=1)
        Xqueue = []
        Yqueue = []

    shouldBeRandom = random.random() < RANDOM_CHANCE or steps <= RANDOM_INTIAL_STEPS
    if shouldBeRandom:
        bestScoreInc = 0
        bestMove: int = None
        bestMoveMat: MutSqMat = None
        for i in range(GAMES):
            game: Game = frame.all_games[i]
            scoreBefore = game.score
            matBefore = game.mat.copy()
            moveID = random.randint(0,3)
            if moveID == 0:
                game.up()
            elif moveID == 1:
                game.down()
            elif moveID == 2:
                game.left()
            elif moveID == 3:
                game.right()
            frame.all_gamevis[i].update_vis()
            game.place_two()
            frame.all_gamevis[i].update_vis()
            scoreAfter = game.score
            scoreDiff = scoreAfter - scoreBefore
            if scoreDiff > bestScoreInc:
                bestScoreInc = scoreDiff
                bestMove = moveID
                bestMoveMat = matBefore
        if bestMove is not None:
            x1 = toTrainingFormat(bestMoveMat)
            y1 = encodeMove(bestMove)
            print("Training Pair Generated",x1,y1)
            Xqueue.append(x1)
            Yqueue.append(y1)
    else:
        pass

    frame.after(50,trainLoop)

def runTraining():
    print("Training Status: building model")
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', input_shape=(GAME_WIDTH,GAME_WIDTH)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(GAME_WIDTH, activation='sigmoid'))
    model.add(layers.Dense(GAME_WIDTH, activation='relu'))

    model.compile(optimizer='adam',metrics=['accuracy'], loss="mean_squared_error")

    frame.after(1500,trainLoop)

frame.master.protocol("WM_DELETE_WINDOW", on_closing)
frame.after_idle(runTraining)

while True:
    frame.update()
    frame.update_idletasks()
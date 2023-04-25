from Deck import Deck
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import cv2
from numba import njit
import random
import os

VALUES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, -2, 26])
CONVERSION = {'King': 0, 'Ace': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'Joker': 13, 'Blank': 14}
CONVERSION_INV = {0: 'King', 1: 'Ace', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: 'Jack', 12: 'Queen', 13: 'Joker', 14: 'Blank'}

models_dir = 'models/RandomvsSelf_4'

@njit(cache=True)
def get_score(grid):
    score = 0
    for row in grid:
        for card in row:
            score += card
    nums = grid[0]
    nscore = score
    for i in range(3):
        if (grid[:, i] == nums[i]).all() and nums[0] != VALUES[14]:
            nscore -= 3 * nums[i]
    if nscore != score:
        return nscore
    nums = grid[:, 0]
    for i in range(3):
        if (grid[i] == nums[i]).all() and nums[0] != VALUES[14]:
            score -= 3 * nums[i]
    return score


@njit(cache=True)
def check_grid_for_more_than_one_unknown(grid):
    x = y = 0
    for i in range(9):
        i, j = i // 3, i % 3
        if grid[i, j] == 14:
            if x and y:
                return [x-1, y-1]
            x = i + 1
            y = j + 1


class Player(gym.Env):
    def __init__(self, id, prevPlayer, nextPlayer, deck, ParentGame, decks=2, name=None):
        super(Player, self).__init__()
        self.prevPlayer = prevPlayer
        self.nextPlayer = nextPlayer
        self.ParentGame = ParentGame
        self.card_in_hand = None
        self.deck = deck
        self.name = name
        self.id = id
        self.grid = np.zeros((3, 3), dtype=np.int32)+14
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Dict({'Card in Hand': spaces.Discrete(15),
                                              "Remaining Cards in Deck": spaces.Box(low=0, high=4*decks, shape=(14,), dtype=np.int32),
                                              'Table': spaces.Box(low=0, high=14, shape=(27,), dtype=np.int32)})
        models = [int(file[:-4]) for file in os.listdir(models_dir)]
        models.sort()
        self.model = PPO.load(f'{models_dir}/{models[-1]}.zip', self, verbose=1, device=1)

    def get_score(self):
        grid = self.grid.copy()
        for i in range(3):
            for j in range(3):
                grid[i][j] = VALUES[grid[i][j]]
        score = get_score(grid)
        return score

    def get_observation(self):
        self.draw_card()
        deck = np.array(list(self.deck.cards.values()), dtype=np.int32)
        observation = np.empty((3, 3, 3), dtype=np.int32)
        observation[0] = self.prevPlayer.grid
        observation[1] = self.grid
        observation[2] = self.nextPlayer.grid
        observation = observation.flatten()
        return {'Card in Hand': CONVERSION[self.card_in_hand],
                'Remaining Cards in Deck': deck,
                'Table': observation}

    def draw_card(self):
        if not len(self.ParentGame.deck):
            if not self.ParentGame.discardPile:
                return None
            self.ParentGame.deck.shuffleBackDiscard(self.ParentGame.discardPile)
            self.ParentGame.discardPile = [self.ParentGame.deck.selectRandomCard()]
            return self.draw_card()
        if not self.ParentGame.discardPile:
            self.card_in_hand = self.ParentGame.deck.selectRandomCard()
            self.ParentGame.deck.removeCard(self.card_in_hand)
            self.deck.removeCard(self.card_in_hand)
            return self.card_in_hand
        discard = self.ParentGame.discardPile[-1]
        if discard in ('Joker', 'King'):
            self.card_in_hand = self.ParentGame.discardPile.pop()
            return discard
        discard = CONVERSION[discard]
        if .85 * np.exp(-.85 * (VALUES[discard] - .5)) > random.random():
            self.card_in_hand = CONVERSION_INV[discard]
            return CONVERSION_INV[discard]
        prev_score = self.get_score()
        for i in range(3):
            for j in range(3):
                prev_num = self.grid[i][j]
                self.grid[i][j] = discard
                if self.get_score() < prev_score - VALUES[14]:
                    self.grid[i][j] = prev_num
                    self.card_in_hand = CONVERSION_INV[discard]
                    return CONVERSION_INV[discard]
                self.grid[i][j] = prev_num

        self.card_in_hand = self.ParentGame.deck.selectRandomCard()
        self.ParentGame.deck.removeCard(self.card_in_hand)
        self.deck.removeCard(self.card_in_hand)
        return self.card_in_hand

    def step(self, action, realPlayer=None, recursive=True):
        if action == 9:
            self.ParentGame.discardPile.append(self.card_in_hand)
            # Flip over a card to reveal its value
            i = check_grid_for_more_than_one_unknown(self.grid)
            if i is not None:
                i, j = i
                self.grid[i, j] = CONVERSION[self.ParentGame.playerCards[self.id][i][j]]
        else:
            prev_card = self.ParentGame.playerCards[self.id][action//3][action % 3]
            if self.grid[action//3, action % 3] == 14:
                self.deck.removeCard(prev_card)
            self.ParentGame.playerCards[self.id][action//3][action % 3] = self.card_in_hand
            self.grid[action//3, action % 3] = CONVERSION[self.card_in_hand]
            self.ParentGame.discardPile.append(prev_card)
            if not (self.grid == 14).any() and self.ParentGame.flippedLast is None:
                self.ParentGame.flippedLast = self.id
        if self.ParentGame.flippedLast is not None:
            self.ParentGame.finalTurn[self.id] = True
        if recursive:
            numPlayers = self.ParentGame.numPlayers
            for i in range(1, numPlayers):
                if self.ParentGame.finalTurn.all():
                    break
                player = self.ParentGame.players[(self.id + i) % numPlayers]
                observation = player.get_observation()
                if realPlayer == player.id and self.ParentGame.flippedLast != player.id:
                    player.render(recursive=True)
                    cv2.waitKey(1)
                    try:
                        action = int(input('Enter Action: '))
                    except:
                        action = int(input('Invalid response. Try again!\nEnter Action: '))
                else:
                    # action = self.action_space.sample()
                    action, _ = player.model.predict(observation)
                player.step(action, recursive=False)
            if self.ParentGame.finalTurn.all():
                scores = {}
                for player in self.ParentGame.players:
                    # Flip each unflipped card over to reveal its value
                    player.grid = [CONVERSION[card] for row in self.ParentGame.playerCards[player.id].copy() for card in row]
                    player.grid = np.array(player.grid).reshape((3, 3))
                    scores[player.id] = player.get_score()
                scores = sorted(scores.items(), key=lambda x: x[1])
                if self.id == scores[0][0]:
                    return self.get_observation(), 1000-self.get_score(), True, {}
                if self.ParentGame.flippedLast == self.id:
                    return self.get_observation(), -(score := self.get_score())*2 - 21*(score <= 10)-1000, True, {}
                return self.get_observation(), -self.get_score()-1000, True, {}
        return self.get_observation(), -self.get_score()/50-2, False, {}

    def reset(self):
        self.ParentGame.reset()
        return self.get_observation()

    def render(self, recursive=True):
        img = np.zeros((396, 297, 3), dtype=np.uint8)
        cards_img = cv2.imread('Cards.png')
        joker = cv2.imread('Joker-Nothing.png')
        img_width = img.shape[1]
        img_height = img.shape[0]
        cards_img = cv2.resize(cards_img, (img_width//3*13, img_height//3*4))
        joker = cv2.resize(joker, (img_width//3*2, img_height//3))
        card_width = cards_img.shape[1]//13
        card_height = cards_img.shape[0]//4
        for i in range(3):
            for j in range(3):
                card = self.grid[i][j]
                if card > 12:
                    card -= 13
                    nimg = joker[:, card_width*card:card_width*(card+1)]
                    img[i*img_height//3:(i+1)*img_height//3, j*img_width//3:(j+1)*img_width//3] = nimg
                    continue
                card = (card-1) % 13
                rand_type = random.randint(0, 3)
                nimg = cards_img[card_height*rand_type:card_height*(rand_type+1), card_width*(card):card_width*(card+1)]
                img[i*img_height//3:(i+1)*img_height//3, j*img_width//3:(j+1)*img_width//3] = nimg
        cv2.imshow(f'ID: {self.id}', img)
        if not recursive:
            return
        for player in self.ParentGame.players:
            if player.id != self.id:
                player.render(recursive=False)
        card = CONVERSION[self.ParentGame.discardPile[-1]]
        discard_img = joker.copy()
        discard_img = cv2.resize(discard_img, (img_width//3*2, img_height//3))
        discard_img = cv2.flip(discard_img, 1)
        nimg = cards_img[:card_height, card_width*card:card_width*(card+1)] if card < 13 else joker[:, :card_width]
        discard_img[:, img_width//3:img_width//3*2] = nimg
        cv2.imshow('Discard', discard_img)
        card = CONVERSION[self.card_in_hand]
        ncard = (card-1) % 13
        card_in_hand_img = cards_img[:card_height, card_width*ncard:card_width*(ncard+1)] if card < 13 else joker[:, :card_width]
        cv2.imshow('Card in Hand', card_in_hand_img)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class NineDown():
    def __init__(self, numPlayers=3, decks=2, PlayerNames=[]):
        self.decks = decks
        self.numPlayers = numPlayers
        self.playerKnownDeck = Deck(decks=decks, types=False)
        self.gamesPlayed = 0
        self.players = [Player(i, None, None, deck=self.playerKnownDeck, ParentGame=self, decks=decks, name=PlayerNames[i] if i < len(PlayerNames) else None) for i in range(numPlayers)]
        for i, player in enumerate(self.players):
            player.prevPlayer = self.players[(i-1) % numPlayers]
            player.nextPlayer = self.players[(i+1) % numPlayers]
        self.reset()

    def score(self, player):
        grid = self.playerCards[player].copy()
        for i in range(3):
            for j in range(3):
                grid[i][j] = VALUES[CONVERSION[grid[i][j]]]
        grid = np.array(grid)
        score = get_score(grid)
        return score

    def reset(self):
        self.gamesPlayed += 1
        decks = self.decks  # Number of decks to use
        numPlayers = self.numPlayers  # Number of players
        self.deck = Deck(decks=decks, types=False)
        self.playerKnownDeck = Deck(decks=decks, types=False)
        self.numPlayers = numPlayers
        self.playerCards = []
        self.finalTurn = np.array([False for _ in range(numPlayers)])
        self.flippedLast = None
        # if self.gamesPlayed % 300 == 0:
        #     models = [int(file[:-4]) for file in os.listdir(models_dir)]
        #     models.sort()
        # Add player known deck to every player
        for player in self.players:
            player.deck = self.playerKnownDeck
            player.grid = np.zeros((3, 3), dtype=np.int32) + 14
            # if self.gamesPlayed % 300 == 0:
            #     player.model.set_parameters(f'{models_dir}/{models[-1]}.zip')

        # Add cards to players to keep track of them
        for player in range(len(self.players)):
            self.playerCards.append([])
            for row in range(3):
                self.playerCards[player].append([])
                for _ in range(3):
                    randomCard = self.deck.selectRandomCard()
                    self.deck.removeCard(randomCard)
                    self.playerCards[player][row].append(randomCard)
        # Add cards to discard pile
        self.discardPile = [self.deck.selectRandomCard()]
        self.deck.removeCard(self.discardPile[0])


# game = NineDown()
# game.score(0)
# Player0 = game.players[0]
# Player0.reset()
# Player0.step(3)
# Player0.reset()

import random


class Deck:
    def __init__(self, decks=1, jokers=True, types=True):
        self.cards = {}
        if types:
            types = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
            values = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
            for type in types:
                for value in values:
                    self.cards[value + ' of ' + type] = 1
            if jokers:
                self.cards['Joker'] = 2
            for card in self.cards:
                self.cards[card] *= decks
        else:
            values = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
            for value in values:
                self.cards[value] = 4
            if jokers:
                self.cards['Joker'] = 2
            for card in self.cards:
                self.cards[card] *= decks

    def __len__(self):
        return sum(self.cards.values())

    def selectRandomCard(self):
        cardsToChooseFrom = [card for card in self.cards if self.cards[card] > 0]
        if len(cardsToChooseFrom) == 0:
            return None
        return random.choice(cardsToChooseFrom)

    def removeCard(self, card):
        self.cards[card] -= 1
    
    def shuffleBackDiscard(self, discard):
        for card in discard:
            self.cards[card] += 1

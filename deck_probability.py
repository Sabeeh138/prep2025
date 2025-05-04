suits = ["Hearts", "Diamonds", "Spades", "Clubs"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "King", "Queen", "Ace", "Jack"]

deck = [(rank, suit) for suit in suits for rank in ranks]

red_cards = [card for card in deck if card[1] in ["Hearts", "Diamonds"]]
p_redCard = len(red_cards)/len(deck)
print(f"Probabilty of drawing a red card: {p_redCard}")

heart_cards = [card for card in red_cards if card[1] == "Hearts"]
p_red_heart = len(heart_cards)/len(red_cards)
print(f"Probabilty of drawing a red card which is a heart: {p_red_heart}")

face_cards = [card for card in deck if card[0] in ["King", "Queen", "Ace"]]
diamond_cards = [card for card in face_cards if card[1] == "Diamonds"]
p_diamond_given_face = len(diamond_cards)/len(face_cards)
print(f"Probability of drawing a diamond given face card: {p_diamond_given_face}")

spade_cards = [card for card in face_cards if card[1] == "Spades"]
queen_cards = [card for card in face_cards if card[0] == "Queen"]

spades_or_queens = set(spade_cards + queen_cards)
p_spades_or_queens_given_faceCards = len(spades_or_queens)/len(face_cards)
print(f"Probability of drawing a spade or a queen given a face card: {p_spades_or_queens_given_faceCards}")
import os
import math
import random
import torch
import torch.nn as nn

# =============================
# 1. Load the Dataset and Mappings
# =============================

# Load the five-letter words dataset
cur_dir = os.path.dirname(__file__)
file_path = os.path.join(cur_dir, '../data/5-letter-words.txt')

# Read the words from the file
with open(file_path, 'r') as f:
    words = [line.strip().lower() for line in f if len(line.strip()) == 5]

# Remove duplicates and ensure all words are lowercase letters
words = list(set(words))
words = [word for word in words if word.isalpha() and word.islower()]

print(f"Total words loaded: {len(words)}")

# Create character mappings
letters = set(''.join(words))
letters = sorted(letters)

char_to_index = {char: idx + 1 for idx, char in enumerate(letters)}  # Start indexing from 1
char_to_index['<pad>'] = 0
index_to_char = {idx: char for char, idx in char_to_index.items()}

vocab_size = len(char_to_index)

print(f"Vocabulary size (including '<pad>'): {vocab_size}")

max_seq_length = 150

# =============================
# 2. Define the Model and Load It
# =============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * self.scale + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class WordleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(WordleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)

    def forward(self, src, tgt):
        src_key_padding_mask = (src == 0).bool()
        tgt_key_padding_mask = (tgt == 0).bool()

        src_emb = self.embedding(src)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        output = self.transformer(
            src=src_emb.transpose(0, 1),
            tgt=tgt_emb.transpose(0, 1),
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        output = self.fc_out(output)
        return output.transpose(0, 1)

# Load the model
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = WordleTransformer(vocab_size=vocab_size, d_model=128, nhead=8, num_layers=3).to(device)
model_path = os.path.join(cur_dir, '../models/model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully.")

# =============================
# 3. Helper Functions
# =============================

def encode_word(word, char_to_index):
    return [char_to_index[char] for char in word]

def encode_game_history(history, char_to_index):
    sequence = []
    for guess, feedback in history:
        encoded_guess = encode_word(guess, char_to_index)
        sequence.extend(encoded_guess)
        sequence.extend(feedback)
    return sequence

def get_feedback(guess, target):
    feedback = [0] * 5
    target_chars = list(target)
    guess_chars = list(guess)

    # First pass for greens
    for i in range(5):
        if guess_chars[i] == target_chars[i]:
            feedback[i] = 2  # Green
            target_chars[i] = None  # Remove matched character

    # Second pass for yellows
    for i in range(5):
        if feedback[i] == 0 and guess_chars[i] in target_chars:
            feedback[i] = 1  # Yellow
            target_chars[target_chars.index(guess_chars[i])] = None  # Remove matched character

    return feedback

def filter_words(words_list, guess, feedback):
    filtered_words = []
    for word in words_list:
        if match_feedback(word, guess, feedback):
            filtered_words.append(word)
    return filtered_words

def match_feedback(word, guess, feedback):
    word_chars = list(word)
    guess_chars = list(guess)
    feedback_copy = feedback.copy()
    word_chars_copy = word_chars.copy()

    # First pass for greens
    for i in range(5):
        if feedback[i] == 2:
            if word_chars[i] != guess_chars[i]:
                return False
            word_chars_copy[i] = None  # Mark as matched

    # Second pass for yellows
    for i in range(5):
        if feedback[i] == 1:
            if guess_chars[i] == word_chars[i]:
                return False  # Should not be in the same position
            elif guess_chars[i] not in word_chars_copy:
                return False  # Letter not in word
            else:
                idx = word_chars_copy.index(guess_chars[i])
                word_chars_copy[idx] = None  # Mark as matched

    # Third pass for grays
    for i in range(5):
        if feedback[i] == 0:
            if guess_chars[i] in word_chars_copy:
                return False  # Letter should not be in word
    return True

# =============================
# 4. Testing Application
# =============================

def predict_next_guess(model, game_history, possible_words, char_to_index, index_to_char, max_seq_length, device):
    model.eval()
    with torch.no_grad():
        input_sequence = encode_game_history(game_history, char_to_index)
        input_sequence = input_sequence[-max_seq_length:]
        input_len = len(input_sequence)
        if input_len < max_seq_length:
            input_sequence = [char_to_index['<pad>']] * (max_seq_length - input_len) + input_sequence
        input_seq_tensor = torch.tensor([input_sequence], dtype=torch.long).to(device)

        # Start token for target sequence
        tgt_input = torch.tensor([[char_to_index['<pad>']]], dtype=torch.long).to(device)

        predicted_word = ''
        for _ in range(5):  # Max length of the word
            output = model(input_seq_tensor, tgt_input)
            # Get probabilities
            probs = nn.functional.softmax(output[:, -1, :], dim=-1)
            # Mask out tokens not corresponding to letters (e.g., pad token)
            probs[0, char_to_index['<pad>']] = 0.0  # Zero out the probability for the padding token
            # Get the top candidate tokens
            k = vocab_size - 1  # Exclude the padding token
            topk = torch.topk(probs, k=k, dim=-1)
            candidate_indices = topk.indices[0].cpu().numpy()
            candidate_probs = topk.values[0].cpu().numpy()

            # Generate possible next characters
            found_word = None
            for idx in candidate_indices:
                char = index_to_char.get(idx, '')
                temp_word = predicted_word + char
                if len(temp_word) == 5:
                    if temp_word in possible_words:
                        predicted_word = temp_word
                        return predicted_word
                elif len(temp_word) < 5:
                    predicted_word += char
                    tgt_input = torch.cat([tgt_input, torch.tensor([[idx]], dtype=torch.long).to(device)], dim=1)
                    break  # Break to predict the next character
            else:
                # If no valid character found, pick a random word from possible_words
                predicted_word = random.choice(possible_words)
                return predicted_word[:5]
        return predicted_word[:5]

def play_wordle(model, target_word, words, char_to_index, index_to_char, max_seq_length, device):
    game_history = []
    attempts = 0
    max_attempts = 6
    possible_words = words.copy()

    print(f"\nStarting a new game. The target word is '{target_word}'.")
    while attempts < max_attempts:
        if attempts == 0:
            guess = 'raise'  # Use a common starting word
        else:
            guess = predict_next_guess(model, game_history, possible_words, char_to_index, index_to_char, max_seq_length, device)
            if guess not in possible_words or len(guess) != 5:
                # If the model predicts an invalid word, pick a random one from possible words
                guess = random.choice(possible_words)

        feedback = get_feedback(guess, target_word)
        game_history.append((guess, feedback))
        print(f"Attempt {attempts+1}: Guess='{guess}', Feedback={feedback}")
        if guess == target_word:
            print("The model guessed the word correctly!")
            return True
        # Update possible words based on feedback
        possible_words = filter_words(possible_words, guess, feedback)
        if not possible_words:
            break
        attempts += 1
    print(f"The model failed to guess the word. The correct word was '{target_word}'.")
    return False

# =============================
# 5. Run the Testing Application
# =============================

# List of target words to test
test_words = ['apple', 'house', 'train', 'water', 'plane']

for target_word in test_words:
    if target_word not in words:
        print(f"Warning: The target word '{target_word}' is not in the word list.")
        continue
    play_wordle(model, target_word, words, char_to_index, index_to_char, max_seq_length, device)

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

cur_dir = os.path.dirname(__file__)

# Helper function to encode a word using raw ASCII values (no restriction to 0-25)
def encode_word(word):
    return [ord(char) for char in word]  # Keeps ASCII values directly (e.g., 'a' -> 97)

# Neural network model for Wordle using LSTM
class WordleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers):
        super().__init__()
        # Embedding layer to map ASCII values to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)  # Convert ASCII values to dense vectors
        result, _ = self.lstm(embedded)
        result = self.dropout(result)
        return self.fc1(result)

def get_feedback(guess, target):
    """
    Compare guess with the target word and return feedback.
    1 = correct, 0 = misplaced, -1 = incorrect
    """
    feedback = torch.full_like(guess, -1)  # Initialize feedback to -1 (incorrect)
    target_counts = torch.zeros(128, dtype=torch.int32)  # To count letters in the target
    
    # Step 1: Handle correct letters
    for i in range(target.shape[1]):
        if guess[0, i] == target[0, i]:
            feedback[0, i] = 1  # Correct letter in the correct place
        else:
            target_counts[target[0, i]] += 1  # Count letters not in the correct place

    # Step 2: Handle misplaced letters
    for i in range(target.shape[1]):
        if feedback[0, i] == -1 and target_counts[guess[0, i]] > 0:  # Misplaced
            feedback[0, i] = 0
            target_counts[guess[0, i]] -= 1  # Decrease the count of that letter in the target
    
    return feedback


def apply_feedback(guess, feedback):
    """
    Modify the guess based on the feedback.
    Letters with feedback 1 (correct) should stay the same.
    Letters with feedback 0 (misplaced) can be re-evaluated (left unchanged for now).
    Letters with feedback -1 (incorrect) should be replaced.
    """
    new_guess = guess.clone()
    
    # For now, we'll just keep the correct guesses, leave the misplaced and incorrect ones for the model to adjust
    for i in range(guess.shape[1]):
        if feedback[0, i] == -1:
            new_guess[0, i] = ord('_')  # Placeholder for incorrect guesses
    
    return new_guess


def train(model, train_data, epochs, device):
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            target_word_tensor = torch.tensor([encode_word(word) for word in batch], dtype=torch.long).to(device)
            optimizer.zero_grad()

            # Initialize blank guess state (e.g., ASCII for '_')
            guess_state = torch.full(target_word_tensor.shape, ord('_'), dtype=torch.long).to(device)

            # Simulate up to 6 guesses
            for _ in range(6):
                predictions = model(guess_state)  # Get predictions for current guess
                
                # Get the guess based on the predictions (argmax for most likely characters)
                predicted_indices = predictions.argmax(dim=2)
                guess_state = predicted_indices

                # Compare guess with target and get feedback (1 = correct, 0 = misplaced, -1 = incorrect)
                feedback = get_feedback(guess_state, target_word_tensor)

                # Adjust guess based on feedback
                guess_state = apply_feedback(guess_state, feedback)

                # If the guess is correct, stop guessing
                if torch.equal(predicted_indices, target_word_tensor):
                    break

            # Calculate loss on final guess
            loss = criterion(predictions.view(-1, 128), target_word_tensor.view(-1))

            # Backpropagation and optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


def test(model, test_data, device):
    model.eval()  # Set the model to evaluation mode
    correct_guesses = 0

    for word in test_data:
        target_word_tensor = torch.tensor(encode_word(word), dtype=torch.long).unsqueeze(0).to(device)

        guess_state = torch.full(target_word_tensor.shape, ord('_'), dtype=torch.long).to(device)

        for _ in range(6):
            # Model makes a guess
            with torch.no_grad():
                predictions = model(guess_state)

            # Get the model's predicted word (use argmax for most likely characters)
            predicted_indices = predictions.argmax(dim=2)
            guess_state = predicted_indices

            # Compare guess with target and get feedback
            feedback = get_feedback(guess_state, target_word_tensor)

            # Adjust guess based on feedback
            guess_state = apply_feedback(guess_state, feedback)

            # Stop if the guess is correct
            if torch.equal(predicted_indices, target_word_tensor):
                break

        # Final check
        predicted_word = ''.join([chr(idx) for idx in guess_state.squeeze(0).cpu().tolist()])
        if predicted_word == word:
            correct_guesses += 1

        # Output the actual answer and the model's guess
        print(f"Answer: {word} | Guess: {predicted_word}")

    # Output accuracy
    print(f"Test Accuracy: {correct_guesses / len(test_data) * 100:.2f}%")

# Function to read the dataset of 5-letter words
def get_data():
    text_path = os.path.join(cur_dir, "../data/5-letter-words.txt")
    words_df = pd.read_csv(text_path, header=None)

    if words_df.empty:
        print("Error: The dataset is empty.")
        return []
    
    return words_df[0].tolist()

# Main function to start training and testing
def main():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset and split into training and test sets (80% train, 20% test)
    dataset = get_data()
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Initialize model
    model = WordleModel(vocab_size=128, embedding_dim=100, hidden_size=512, output_size=128, num_layers=1)
    model.to(device)
    
    # Train the model
    train(model, train_data, epochs=20, device=device)
    
    # Test the model and print results
    test(model, test_data, device=device)
    
    model_dir = os.path.join(cur_dir, '../models/model.pth')

    torch.save(model.state_dict(), model_dir)

if __name__ == "__main__":
    main()

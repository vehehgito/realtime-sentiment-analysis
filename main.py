# Importing NLTK and VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
import nltk
from nltk.sentiment import vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Importing Gooey for GUI
from gooey import Gooey, GooeyParser

# Invoking the Gooey Decorator Function
@Gooey(program_name="Sentiment Analysis", 
       program_description="Analyze the sentiment of given text in real-time", 
       header_show_title="Real-time Sentiment Analysis Chatbot", 
       header_text='Sentiment Analysis', 
       header_bg_color='#168A43')

# Main Function
def main():
    # Create a SentimentIntensityAnalyzer object
    analyzer = vader.SentimentIntensityAnalyzer()

    # Create a GooeyParser object for taking user input
    parser = GooeyParser()

    # Get input from the user
    parser.add_argument("TextVar", help="Hello! How are you doing? Feel free to talk about something and I will reply accordingly.", widget="Textarea")
    args = parser.parse_args()

    # Perform sentiment analysis
    sentiment_scores = analyzer.polarity_scores(args.TextVar)

    # Reply logic
    compound = sentiment_scores['compound']
    pos = sentiment_scores['pos']
    neg = sentiment_scores['neg']

    if compound >= 0.05 and pos > neg:
        reply = "Positive"
    elif compound <= -0.05 and neg > pos:
        reply = "Negative"
    else:
        reply = "Neutral"

    # Print the reply
    print("----------------------------------------\n")
    print(reply,"\n")
    print("----------------------------------------\n")


    # Print the sentiment scores
    print("Upon analysis, we obtain the following data:")
    for key, value in sentiment_scores.items():
        print(f"{key}: {value}")


main()
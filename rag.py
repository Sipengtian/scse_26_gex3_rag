import sys

from agent import answer_question


DEFAULT_QUESTION = (
    "What exact phone number should I call for the University IT Service Desk?"
)


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = DEFAULT_QUESTION

    print("QUESTION:")
    print(question)
    print()
    print("ANSWER:")
    print(answer_question(question))


if __name__ == "__main__":
    main()

from search import search_prompt


def main() -> None:
    print("Faça sua pergunta (CTRL+C para sair):")
    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue

        answer = search_prompt(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()

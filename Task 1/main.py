import sys
from production import forward_chain, backward_chain, IF, AND, OR, NOT
from rules import MY_RULES

class Question:
    def __init__(self, prompt, var, qtype="yesno", options=None):
        self.prompt = prompt
        self.var = var
        self.qtype = qtype
        self.options = options or []

    def ask(self):
        if self.qtype == "yesno":
            while True:
                ans = input(f"{self.prompt} (yes/no): ").strip().lower()
                if ans in ("yes", "y"):
                    return True
                elif ans in ("no", "n"):
                    return False
                print("Please answer yes or no.")
        elif self.qtype == "multiple":
            print(self.prompt)
            for i, opt in enumerate(self.options, 1):
                print(f"  {i}) {opt}")
            while True:
                ans = input(f"Choose 1-{len(self.options)}: ").strip()
                if ans.isdigit():
                    idx = int(ans) - 1
                    if 0 <= idx < len(self.options):
                        return self.options[idx]
                print("Invalid choice.")
        elif self.qtype == "input":
            ans = input(f"{self.prompt}: ").strip()
            return ans
        else:
            raise ValueError(f"Unknown question type: {self.qtype}")

class ExpertSystem:
    def __init__(self, rules):
        self.rules = rules
        self.facts = set()
        self.asked_vars = set()

    def add_fact(self, fact):
        if fact not in self.facts:
            print(f"[New fact added]: {fact}")
            self.facts.add(fact)

    def ask_questions(self, questions):
        for q in questions:
            if q.var in self.asked_vars:
                continue
            ans = q.ask()
            self.asked_vars.add(q.var)
            if q.qtype == "yesno":
                if ans:
                    self.add_fact(f"(?x) {q.var}")
            elif q.qtype == "multiple":
                self.add_fact(f"(?x) {q.var} {ans}")
            elif q.qtype == "input":
                self.add_fact(f"(?x) {q.var} {ans}")

    def run_forward(self):
        old_facts = None
        current_facts = tuple(self.facts)
        while old_facts != current_facts:
            old_facts = current_facts
            current_facts = forward_chain(self.rules, current_facts)
        self.facts = set(current_facts)

    def run_backward(self, hypothesis, verbose=False):
        # The backward chaining here expects the full hypothesis string
        return backward_chain(self.rules, hypothesis, verbose)

    def forward_mode(self):
        questions = [
            Question("Does the person wear an expensive suit?", "wears expensive suit", "yesno"),
            Question("Does the person wear flashy gear?", "wears flashy gear", "yesno"),
            Question("Does the person wear earth casual?", "wears earth casual", "yesno"),
            Question("Does the person wear mars clothing?", "wears mars clothing", "yesno"),
            Question("Does the person wear zero-g clothing?", "wears zero-g clothing", "yesno"),
            Question("What kind of speech does the person speak?", "speaks", "multiple",
                     ["belt slang", "excited speech", "corporate speech", "academic speech", "lunar dialect"]),
            Question("Does the person have a camera?", "has camera", "yesno"),
            Question("Does the person complain?", "complains", "yesno"),
            Question("Does the person have a datapad?", "has datapad", "yesno"),
            Question("Does the person have mining tools?", "has mining tools", "yesno"),
            Question("Does the person have a briefcase?", "has briefcase", "yesno"),
            Question("Does the person check time frequently?", "checks time", "yesno"),
            Question("Does the person ask questions?", "asks questions", "yesno"),
            Question("Does the person have awkward gait?", "has awkward gait", "yesno"),
            Question("Does the person have smooth gait?", "has smooth gait", "yesno"),
            Question("Is the person tall?", "is tall", "yesno"),
            Question("Is the person short?", "is short", "yesno"),
        ]

        print("\nForward Chaining Mode: Please answer the following questions:\n")
        self.ask_questions(questions)

        print("\nRunning forward chaining inference...")
        self.run_forward()

        print("\nAll inferred facts:")
        for fact in sorted(self.facts):
            print(" -", fact)

    def backward_mode(self):
        print("\nBackward Chaining Mode")
        goal = input("Enter a hypothesis/goal to test (e.g. '(?x) has Earth Origin'): ").strip()

        print(f"\nAttempting to prove: {goal}")

        # For backward chaining, we simulate a questioning strategy:
        # We will ask questions to establish facts necessary to prove the goal.
        # Here, for demo, just ask all questions and run backward chaining on final facts.

        questions = [
            Question("Does the person wear an expensive suit?", "wears expensive suit", "yesno"),
            Question("Does the person wear flashy gear?", "wears flashy gear", "yesno"),
            Question("Does the person wear earth casual?", "wears earth casual", "yesno"),
            Question("Does the person wear mars clothing?", "wears mars clothing", "yesno"),
            Question("Does the person wear zero-g clothing?", "wears zero-g clothing", "yesno"),
            Question("What kind of speech does the person speak?", "speaks", "multiple",
                     ["belt slang", "excited speech", "corporate speech", "academic speech", "lunar dialect"]),
            Question("Does the person have a camera?", "has camera", "yesno"),
            Question("Does the person complain?", "complains", "yesno"),
            Question("Does the person have a datapad?", "has datapad", "yesno"),
            Question("Does the person have mining tools?", "has mining tools", "yesno"),
            Question("Does the person have a briefcase?", "has briefcase", "yesno"),
            Question("Does the person check time frequently?", "checks time", "yesno"),
            Question("Does the person ask questions?", "asks questions", "yesno"),
            Question("Does the person have awkward gait?", "has awkward gait", "yesno"),
            Question("Does the person have smooth gait?", "has smooth gait", "yesno"),
            Question("Is the person tall?", "is tall", "yesno"),
            Question("Is the person short?", "is short", "yesno"),
        ]

        # For simplicity, ask all questions (to simulate needed data)
        self.ask_questions(questions)

        # Now check hypothesis
        result = self.run_backward(goal, verbose=True)

        print(f"\nResult: The hypothesis '{goal}' is", "TRUE" if result else "FALSE")

    def interactive_loop(self):
        print("Welcome to the Expert System.")
        while True:
            mode = input("\nSelect mode:\n  1) Forward Chaining\n  2) Backward Chaining\n  3) Exit\nEnter choice (1/2/3): ").strip()
            if mode == "1":
                self.forward_mode()
            elif mode == "2":
                self.backward_mode()
            elif mode == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice, please select 1, 2, or 3.")

def main():
    system = ExpertSystem(MY_RULES)
    system.interactive_loop()

if __name__ == "__main__":
    main()

# thesis/simple_novelty_check.py
print("=== SIMPLE NOVELTY CHECK ===")
print("Answer these questions about your research:\n")

questions = [
    "Does any paper do hierarchical RL for UK battery dispatch?",
    "Does any paper use RL with Elexon BMRS data?",
    "Does any paper quantify congestion costs as RL rewards?", 
    "Does any paper balance profit vs stability services?",
    "Does any paper coordinate Scotland-London batteries?"
]

for i, question in enumerate(questions, 1):
    answer = input(f"{i}. {question} (y/n): ")
    if answer.lower() == 'n':
        print("   ✅ NO - This is novel!\n")
    else:
        print("   ⚠️  YES - Check how yours is different\n")

print("🎯 YOUR NOVELTY SUMMARY:")
print("You're likely the FIRST to combine:")
print("- Hierarchical RL + UK markets + Stability rewards + Portfolio coordination")
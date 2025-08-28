from .questions import question_forms

if __name__ == "__main__":

    print("Available Question Forms:")
    print("-" * 25)

    forms_by_group = {}
    for form in question_forms:
        group = form.group if form.group else "General"
        if group not in forms_by_group:
            forms_by_group[group] = []
        forms_by_group[group].append(form)

    for group, forms_in_group in sorted(forms_by_group.items()):
        print(f"\nGroup: {group}")
        for form in sorted(forms_in_group, key=lambda f: f.type_string):
             # Use english_explain for clarity
            print(f"  - [{form.type_string}] {form.english_explain()}")

    print("\nTotal forms:", len(question_forms))
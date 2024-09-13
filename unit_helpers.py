def display_result(result):
    if result == False:
        print('Failed')
    else:
        print(result)

def detailed_response(issue_list):
    sorted_issues = sorted(issue_list, key=lambda x: x[0])
    return sorted_issues

def simplify_response(issue_list):
    sorted_issues = sorted(issue_list, key=lambda x: x[0])
    seen_lines = set()
    unique_line_numbers = []
    for issue in sorted_issues:
        line_number = issue[0]
        if line_number not in seen_lines:
            unique_line_numbers.append(line_number)
            seen_lines.add(line_number)
    return unique_line_numbers

def pick_response(response, detailed, check):
    if detailed:
        print(check + ': Complete')
        return detailed_response(response)
    elif not detailed:
        print(check + ': Complete')
        return simplify_response(response)
// lib/screens/age_question_flow.dart
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;

class AgeQuestionFlow extends StatefulWidget {
  final int age;
  const AgeQuestionFlow({super.key, required this.age});

  @override
  State<AgeQuestionFlow> createState() => _AgeQuestionFlowState();
}

class _AgeQuestionFlowState extends State<AgeQuestionFlow> {
  late final List<_Q> _questions;
  int _index = 0;
  final Map<int, int> _answers = {}; // questionIndex -> selectedOptionIndex

  @override
  void initState() {
    super.initState();
    _questions = _buildQuestionsForAge(widget.age);
  }

  List<_Q> _buildQuestionsForAge(int age) {
    if (age >= 15 && age <= 30) {
      return [
        _Q(
          question:
              'How often do you use electronic devices (phone/tablet) before bed?',
          options: ['Never', 'Sometimes', 'Often', 'Always'],
        ),
        _Q(
          question: 'Are your sleep/wake times regular?',
          options: ['Yes, regular', 'Sometimes irregular', 'Often irregular'],
        ),
        _Q(
          question:
              'How often do you consume stimulants (coffee, alcohol, tobacco)?',
          options: ['Never', 'Occasionally', 'Regularly'],
        ),
        _Q(
          question:
              'How would you rate your stress level (study/work/emotional)?',
          options: ['Low', 'Moderate', 'High'],
        ),
      ];
    } else if (age >= 31 && age <= 65) {
      return [
        _Q(
          question:
              'Do you have chronic health conditions that may affect sleep?',
          options: ['No', 'Yes — mild', 'Yes — significant'],
        ),
        _Q(
          question:
              'Have you experienced hormonal changes (e.g., menopause symptoms)?',
          options: ['No', 'Yes'],
        ),
        _Q(
          question: 'How often do you have trouble falling or staying asleep?',
          options: ['Rarely', 'Sometimes', 'Often'],
        ),
        _Q(
          question: 'Are you taking medications that may affect sleep?',
          options: ['No', 'Occasionally', 'Yes regularly'],
        ),
      ];
    } else {
      // age > 65
      return [
        _Q(
          question: 'Do you get up at night to urinate (nocturia)?',
          options: ['Never', 'Sometimes', 'Often'],
        ),
        _Q(
          question:
              'Do you have chronic pain or joint problems affecting sleep?',
          options: ['No', 'Yes — mild', 'Yes — severe'],
        ),
        _Q(
          question:
              'Do you experience daytime sleepiness or increased napping?',
          options: ['No', 'Sometimes', 'Yes, frequently'],
        ),
        _Q(
          question:
              'Do psychosocial issues (loneliness, low mood) affect your sleep?',
          options: ['No', 'Sometimes', 'Yes'],
        ),
      ];
    }
  }

  void _selectOption(int optionIdx) {
    setState(() {
      _answers[_index] = optionIdx;
    });
  }

  Future<void> _onNext() async {
    if (!_answers.containsKey(_index)) return; // guard: must choose
    if (_index < _questions.length - 1) {
      setState(() => _index++);
      return;
    }

    // last question -> save results
    final prefs = await SharedPreferences.getInstance();
    final gender = prefs.getString('profile_gender') ?? 'unknown';

    final Map<String, dynamic> payload = {
      'gender': gender,
      'age': widget.age,
      'group': widget.age >= 15 && widget.age <= 30
          ? '15-30'
          : (widget.age >= 31 && widget.age <= 65 ? '31-65' : '65+'),
      'answers': _answers.map((qIdx, optIdx) => MapEntry(
            _questions[qIdx].question,
            _questions[qIdx].options[optIdx],
          )),
    };

    await prefs.setString('profile_age_questionnaire', jsonEncode(payload));

    final uri =
        Uri.parse('http://10.0.2.2:8000/submit_survey'); // hoặc IP server thật
    await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    );
    // also set a simpler key for quick access
    await prefs.setString('profile_age_group', payload['group']);
    await prefs.setBool('seenOnboarding', true);

    if (!mounted) return;
    Navigator.pushReplacementNamed(context, '/home');
  }

  void _onBack() {
    if (_index == 0) {
      Navigator.pop(context);
      return;
    }
    setState(() => _index--);
  }

  @override
  Widget build(BuildContext context) {
    final q = _questions[_index];
    final selected = _answers[_index];

    return Scaffold(
      body: Container(
        padding: const EdgeInsets.fromLTRB(20, 40, 20, 20),
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF141328), Color(0xFF2A2140)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // header with back and progress
              Row(
                children: [
                  IconButton(
                    onPressed: _onBack,
                    icon: const Icon(Icons.arrow_back, color: Colors.white),
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      'Question ${_index + 1} of ${_questions.length}',
                      style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 16,
                          fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                q.question,
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 18),

              // options (like the gender screen: tappable cards)
              ...List.generate(q.options.length, (i) {
                final opt = q.options[i];
                final isSelected = selected == i;
                return GestureDetector(
                  onTap: () => _selectOption(i),
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: isSelected ? Colors.white12 : Colors.transparent,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: isSelected ? Colors.white : Colors.white12,
                        width: isSelected ? 2 : 1,
                      ),
                    ),
                    child: Text(
                      opt,
                      style: TextStyle(
                        color: isSelected ? Colors.white : Colors.white70,
                        fontSize: 16,
                        fontWeight:
                            isSelected ? FontWeight.w700 : FontWeight.w500,
                      ),
                    ),
                  ),
                );
              }),

              const Spacer(),

              // bottom actions
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: _index == 0 ? null : _onBack,
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        side: BorderSide(color: Colors.white12),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                      child: const Text('Back',
                          style: TextStyle(color: Colors.white70)),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: _answers.containsKey(_index) ? _onNext : null,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                      child: Text(
                          _index == _questions.length - 1 ? 'Finish' : 'Next'),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Simple data holder for question + options
class _Q {
  final String question;
  final List<String> options;
  _Q({required this.question, required this.options});
}

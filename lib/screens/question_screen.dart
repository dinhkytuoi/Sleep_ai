// lib/screens/question_screen.dart
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class QuestionScreen extends StatefulWidget {
  const QuestionScreen({super.key});

  @override
  State<QuestionScreen> createState() => _QuestionScreenState();
}

class _QuestionScreenState extends State<QuestionScreen> {
  String? _selected;

  // Lưu gender vào SharedPreferences và điều hướng sang question_age
  Future<void> _onNext() async {
    if (_selected == null) {
      // nếu chưa chọn thì show snackbar (cache messenger trước async)
      final messenger = ScaffoldMessenger.of(context);
      messenger.showSnackBar(
        const SnackBar(content: Text('Please select a gender')),
      );
      return;
    }

    // cache navigator before any await to avoid analyzer warnings
    final navigator = Navigator.of(context);

    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('profile_gender', _selected!);

    if (!mounted) return;
    navigator.pushReplacementNamed('/question_age');
  }

  Widget _buildOption(String text) {
    final bool selected = _selected == text;
    return GestureDetector(
      onTap: () => setState(() => _selected = text),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 10),
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: selected ? Colors.white12 : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: selected ? Colors.white : Colors.white12,
            width: selected ? 2 : 1,
          ),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: selected ? Colors.white : Colors.white70,
            fontSize: 18,
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // options giống mock bạn đưa (sửa/đuổi tuỳ ý)
    final options = ['Female', 'Male'];

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
              const SizedBox(height: 6),
              const Text(
                'Sex and hormone levels can influence sleep patterns',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.white),
              ),
              const SizedBox(height: 12),
              const Text(
                'Which option best describes you?',
                style: TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 24),

              // options
              ...options.map(_buildOption),

              const Spacer(),

              // Next button (disabled nếu chưa chọn)
              ElevatedButton(
                onPressed: _selected == null ? null : _onNext,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                ),
                child: const Text('Next'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// lib/screens/question_age_screen.dart
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'age_questionnaire_screen.dart';

class QuestionAgeScreen extends StatefulWidget {
  const QuestionAgeScreen({super.key});

  @override
  State<QuestionAgeScreen> createState() => _QuestionAgeScreenState();
}

class _QuestionAgeScreenState extends State<QuestionAgeScreen> {
  DateTime? _picked;

  Future<void> _pickDate() async {
    final now = DateTime.now();
    final picked = await showDatePicker(
      context: context,
      initialDate: DateTime(now.year - 25),
      firstDate: DateTime(now.year - 100),
      lastDate: now,
      builder: (ctx, child) {
        return Theme(
          data: Theme.of(ctx).copyWith(
            colorScheme: const ColorScheme.dark(
              primary: Color(0xFF3B82F6),
              onPrimary: Colors.white,
              surface: Color(0xFF222222),
              onSurface: Colors.white,
            ),
            dialogTheme: const DialogThemeData(
              backgroundColor: Color(0xFF111111),
            ),
          ),
          child: child ?? const SizedBox(),
        );
      },
    );

    if (picked != null) {
      if (!mounted) return;
      setState(() {
        _picked = picked;
      });
    }
  }

  // Sửa _finish: lưu birthday/age rồi chuyển sang luồng câu hỏi theo tuổi
  Future<void> _finish() async {
    if (_picked == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please pick your birthday')),
      );
      return;
    }

    final prefs = await SharedPreferences.getInstance();

    // compute age
    final now = DateTime.now();
    int age = now.year - _picked!.year;
    if (now.month < _picked!.month ||
        (now.month == _picked!.month && now.day < _picked!.day)) {
      age--;
    }

    // save to profile basic info
    await prefs.setString('profile_birthday', _picked!.toIso8601String());
    await prefs.setInt('profile_age', age);
    await prefs.setBool('seenOnboarding', true);

    if (!mounted) return;

    // Chuyển sang màn hỏi theo tuổi (AgeQuestionFlow)
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => AgeQuestionFlow(age: age)),
    );
  }

  @override
  Widget build(BuildContext context) {
    final label = _picked == null
        ? 'Pick your birthday'
        : DateFormat('yyyy-MM-dd').format(_picked!);

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
            children: [
              const SizedBox(height: 8),
              const Text(
                'As we age, our sleep needs and challenges change',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.white),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              const Text(
                'Please select your birthday.',
                style: TextStyle(color: Colors.white70),
              ),
              const SizedBox(height: 28),
              GestureDetector(
                onTap: _pickDate,
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white12, width: 1.4),
                  ),
                  child: Row(
                    children: [
                      Text(
                        label,
                        style:
                            const TextStyle(color: Colors.white, fontSize: 16),
                      ),
                      const Spacer(),
                      const Icon(Icons.calendar_month, color: Colors.white70),
                    ],
                  ),
                ),
              ),
              const Spacer(),
              SizedBox(
                width: double.infinity, // full width
                child: ElevatedButton(
                  onPressed: _finish,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text('Next'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

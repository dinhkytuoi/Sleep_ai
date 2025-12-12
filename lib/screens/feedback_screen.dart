// lib/screens/feedback_screen.dart
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class FeedbackScreen extends StatefulWidget {
  const FeedbackScreen({super.key});

  @override
  State<FeedbackScreen> createState() => _FeedbackScreenState();
}

class _FeedbackScreenState extends State<FeedbackScreen> {
  final _ctrl = TextEditingController();

  Future<void> _send() async {
    final text = _ctrl.text.trim();
    if (text.isEmpty) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text('Please enter feedback')));
      return;
    }
    final prefs = await SharedPreferences.getInstance();
    final items = prefs.getStringList('feedback_items') ?? [];
    items.add(
        jsonEncode({'text': text, 'ts': DateTime.now().toIso8601String()}));
    await prefs.setStringList('feedback_items', items);
    ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Feedback saved. Thank you!')));
    Navigator.pop(context);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Feedback'),
        backgroundColor: Colors.transparent,
        elevation: 0,
      ),
      body: Container(
        padding: const EdgeInsets.all(18),
        decoration: const BoxDecoration(
          gradient:
              LinearGradient(colors: [Color(0xFF141328), Color(0xFF2A2140)]),
        ),
        child: Column(
          children: [
            TextField(
              controller: _ctrl,
              minLines: 4,
              maxLines: 8,
              style: const TextStyle(color: Colors.white),
              decoration: const InputDecoration(
                hintText: 'Tell us what you think...',
                hintStyle: TextStyle(color: Colors.white54),
                filled: true,
                fillColor: Color(0xFF1F1B2A),
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.all(Radius.circular(10))),
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton(
                onPressed: _send, child: const Text('Send Feedback')),
          ],
        ),
      ),
    );
  }
}

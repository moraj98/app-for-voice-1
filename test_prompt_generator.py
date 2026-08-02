"""Tests for the prompt generator. Run with: python -m unittest -v"""
import unittest

from prompt_generator import (
    detect_intent,
    generate_prompt,
    parse_terms,
    refine_prompt,
    refine_text,
)


class ParseTermsTests(unittest.TestCase):
    def test_sorts_words_into_buckets(self):
        buckets, _ = parse_terms("cat, cyberpunk, neon, close-up, 16:9")
        self.assertEqual(buckets["subject"], ["cat"])
        self.assertEqual(buckets["style"], ["cyberpunk"])
        self.assertEqual(buckets["lighting"], ["neon"])
        self.assertEqual(buckets["camera"], ["close-up"])
        self.assertEqual(buckets["aspect"], ["16:9"])

    def test_negated_words_become_avoid_list(self):
        buckets, negatives = parse_terms("forest, no people, without text")
        self.assertEqual(buckets["subject"], ["forest"])
        self.assertEqual(negatives, ["people", "text"])

    def test_leading_dash_negates(self):
        _, negatives = parse_terms("portrait, -blurry")
        self.assertEqual(negatives, ["blurry"])

    def test_sentence_chunk_is_kept_verbatim(self):
        buckets, _ = parse_terms("why is my deploy slow")
        self.assertEqual(buckets["subject"], ["why is my deploy slow"])

    def test_short_chunk_drops_stopwords(self):
        buckets, _ = parse_terms("a cat, in the rain")
        self.assertEqual(buckets["subject"], ["cat", "rain"])

    def test_measurements_become_length(self):
        buckets, _ = parse_terms("email, 200 words")
        self.assertEqual(buckets["length"], ["200 words"])

    def test_multi_word_phrases_survive_word_splitting(self):
        buckets, _ = parse_terms("desert highway, golden hour, drone shot")
        self.assertEqual(buckets["lighting"], ["golden hour"])
        self.assertEqual(buckets["camera"], ["drone shot"])

    def test_filler_is_stripped(self):
        buckets, _ = parse_terms("please write really good copy")
        self.assertNotIn("please", " ".join(buckets["subject"]))

    def test_duplicates_are_collapsed(self):
        buckets, _ = parse_terms("neon, neon, neon")
        self.assertEqual(buckets["lighting"], ["neon"])

    def test_empty_input_yields_nothing(self):
        buckets, negatives = parse_terms("")
        self.assertEqual((buckets, negatives), ({}, []))


class DetectIntentTests(unittest.TestCase):
    def _intent(self, raw):
        buckets, _ = parse_terms(raw)
        return detect_intent(raw, buckets)[0]

    def test_image_words(self):
        self.assertEqual(self._intent("photo of a lighthouse, watercolor"), "image")

    def test_video_words(self):
        self.assertEqual(self._intent("30 second video of a city, tracking shot"), "video")

    def test_code_words(self):
        self.assertEqual(self._intent("python function to parse csv"), "code")

    def test_writing_words(self):
        self.assertEqual(self._intent("blog post about bees, friendly"), "writing")

    def test_analysis_words(self):
        self.assertEqual(self._intent("compare two pricing plans, evaluate"), "analysis")

    def test_unknown_falls_back_to_general(self):
        self.assertEqual(self._intent("kettle"), "general")


class GeneratePromptTests(unittest.TestCase):
    def test_image_prompt_has_visual_sections(self):
        result = generate_prompt("cat, cyberpunk, neon, close-up, 16:9, no people")
        self.assertEqual(result["intent"], "image")
        self.assertIn("A cyberpunk image of cat", result["prompt"])
        self.assertIn("Composition: close-up", result["prompt"])
        self.assertIn("Lighting: neon", result["prompt"])
        self.assertIn("Avoid: people", result["prompt"])
        self.assertIn("16:9", result["prompt"])

    def test_video_prompt_has_duration_and_motion(self):
        result = generate_prompt("drone shot over a desert highway, cinematic, video")
        self.assertEqual(result["intent"], "video")
        self.assertIn("Motion:", result["prompt"])
        self.assertIn("Duration:", result["prompt"])

    def test_writing_prompt_has_role_and_slots(self):
        result = generate_prompt("blog post about remote work, beginners, friendly, short")
        self.assertEqual(result["intent"], "writing")
        self.assertTrue(result["prompt"].startswith("You are an experienced writer"))
        self.assertIn("Audience: beginners", result["prompt"])
        self.assertIn("Tone: friendly", result["prompt"])

    def test_task_line_does_not_repeat_the_format(self):
        result = generate_prompt("blog post about remote work")
        self.assertNotIn("blog post about blog post", result["prompt"])

    def test_code_prompt_adds_engineering_constraints(self):
        result = generate_prompt("python flask rate limiter for the api")
        self.assertEqual(result["intent"], "code")
        self.assertIn("Stack:", result["prompt"])
        self.assertIn("tests", result["prompt"])

    def test_mood_words_become_tone_in_text_prompts(self):
        result = generate_prompt("voiceover for a product demo, warm, energetic, 30 seconds")
        self.assertIn("Tone:", result["prompt"])
        self.assertIn("Length: 30 seconds", result["prompt"])
        self.assertNotIn("Style notes", result["prompt"])

    def test_forced_intent_overrides_detection(self):
        result = generate_prompt("blog post about bees", intent="image")
        self.assertEqual(result["intent"], "image")
        self.assertEqual(result["detected_intent"], "writing")
        self.assertIn("image of", result["prompt"])

    def test_suggestions_list_missing_slots(self):
        result = generate_prompt("cat", intent="image")
        self.assertTrue(any("style" in s for s in result["suggestions"]))

    def test_filled_slots_are_not_suggested(self):
        result = generate_prompt("cat, watercolor, golden hour, wide shot, 16:9")
        self.assertEqual(result["suggestions"], [])

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            generate_prompt("   ")

    def test_unknown_intent_raises(self):
        with self.assertRaises(ValueError):
            generate_prompt("cat", intent="sculpture")


class RefineTests(unittest.TestCase):
    def test_refine_text_drops_filler_and_repeats(self):
        cleaned = refine_text("Please write a really nice email.\n\n\nPlease write a really nice email.")
        self.assertEqual(cleaned, "write a nice email.")

    def test_refine_prompt_keeps_earlier_choices(self):
        first = generate_prompt("blog post about bees, friendly")
        second = refine_prompt(first["prompt"], "beginners, 500 words")
        self.assertIn("Tone: friendly", second["prompt"])
        self.assertIn("Audience: beginners", second["prompt"])
        self.assertIn("500 words", second["prompt"])

    def test_refine_prompt_is_stable_when_nothing_is_added(self):
        first = generate_prompt("cat, watercolor, golden hour, wide shot, 16:9")
        second = refine_prompt(first["prompt"])
        third = refine_prompt(second["prompt"])
        self.assertEqual(second["prompt"], third["prompt"])

    def test_refine_prompt_can_switch_intent(self):
        first = generate_prompt("cat, watercolor")
        second = refine_prompt(first["prompt"], "video, tracking shot", intent="video")
        self.assertEqual(second["intent"], "video")


if __name__ == "__main__":
    unittest.main()

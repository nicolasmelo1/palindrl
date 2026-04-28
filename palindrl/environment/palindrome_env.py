from __future__ import annotations

import string
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(IntEnum):
    COMPARE = 0
    MOVE_INWARD = 1
    ANSWER_PALINDROME = 2
    ANSWER_NOT_PALINDROME = 3


ACTION_COMPARE = int(Action.COMPARE)
ACTION_MOVE_INWARD = int(Action.MOVE_INWARD)
ACTION_ANSWER_PALINDROME = int(Action.ANSWER_PALINDROME)
ACTION_ANSWER_NOT_PALINDROME = int(Action.ANSWER_NOT_PALINDROME)
ACTION_NAMES = [action.name for action in Action]

OBSERVATION_CHARS = "".join(
    dict.fromkeys(string.ascii_letters + string.digits + string.punctuation + " |=")
)


def build_char_vocab() -> dict[str, int]:
    char_to_id = {"<pad>": 0}
    for idx, ch in enumerate(OBSERVATION_CHARS, start=1):
        char_to_id[ch] = idx
    return char_to_id


def build_observation_vocab() -> dict[str, int]:
    return build_char_vocab()


@dataclass
class EnvState:
    text: str = ""
    normalized_text: str = ""
    is_palindrome: bool = False
    left: int = 0
    right: int = 0
    steps: int = 0
    mismatch_observed: bool = False
    has_compared: bool = False
    ready_to_move: bool = False
    last_compared_pair: tuple[int, int] | None = None


def normalize_text(text: str, ignored_characters: set[str]) -> str:
    return "".join(ch for ch in text if ch not in ignored_characters)


def canonicalize_text(text: str, case_sensitive: bool) -> str:
    if case_sensitive:
        return text
    return text.casefold()


class RandomPalindromeEnv(gym.Env[dict[str, np.ndarray], int]):
    """
    Multi-step palindrome environment.

    State contains the text and two pointers (left/right). The agent must learn
    a procedure:
    - compare current pointer chars
    - move inward when equal
    - answer NOT palindrome on mismatch
    - answer palindrome once pointers cross
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        min_len: int = 3,
        max_len: int = 12,
        palindrome_probability: float = 0.5,
        balanced_sampling: bool = True,
        space_probability: float = 0.5,
        separator_chars: str = " -,.",
        ignore_spaces_for_label: bool = True,
        ignored_characters: str = " " + string.punctuation,
        case_sensitive: bool = False,
        final_correct_reward: float = 1.0,
        final_incorrect_reward: float = -1.0,
        invalid_answer_penalty: float = -0.20,
        repeat_compare_penalty: float = -0.05,
        compare_match_reward: float = 0.05,
        compare_mismatch_reward: float = 0.05,
        move_valid_reward: float = 0.03,
        move_invalid_penalty: float = -0.10,
        step_penalty: float = -0.01,
        timeout_penalty: float = -0.50,
        max_steps: int | None = None,
        max_observation_len: int | None = None,
    ) -> None:
        super().__init__()
        if min_len < 1:
            raise ValueError("min_len must be >= 1")
        if max_len < min_len:
            raise ValueError("max_len must be >= min_len")
        if not 0.0 <= palindrome_probability <= 1.0:
            raise ValueError("palindrome_probability must be between 0 and 1")
        if not 0.0 <= space_probability <= 1.0:
            raise ValueError("space_probability must be between 0 and 1")

        # Sampling config.
        self.min_len = min_len
        self.max_len = max_len
        self.palindrome_probability = palindrome_probability
        self.balanced_sampling = balanced_sampling
        self.space_probability = space_probability
        self.separator_chars = "".join(dict.fromkeys(separator_chars))

        # Normalization config.
        self.ignore_spaces_for_label = ignore_spaces_for_label
        self.case_sensitive = case_sensitive
        ignored = set(ignored_characters)
        if self.ignore_spaces_for_label:
            ignored.add(" ")
        self.ignored_characters = (
            ignored if self.case_sensitive else {ch.casefold() for ch in ignored}
        )

        # Reward config.
        self.final_correct_reward = final_correct_reward
        self.final_incorrect_reward = final_incorrect_reward
        self.invalid_answer_penalty = invalid_answer_penalty
        self.repeat_compare_penalty = repeat_compare_penalty
        self.compare_match_reward = compare_match_reward
        self.compare_mismatch_reward = compare_mismatch_reward
        self.move_valid_reward = move_valid_reward
        self.move_invalid_penalty = move_invalid_penalty
        self.step_penalty = step_penalty
        self.timeout_penalty = timeout_penalty

        # Observation config.
        # Text max length with inserted spaces.
        self.max_text_len = 2 * max_len - 1
        if max_steps is None:
            self.max_steps = 2 * self.max_text_len + 4
        else:
            self.max_steps = max_steps

        self.char_to_id = build_char_vocab()
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        inferred_obs_len = len(
            f"{'a' * self.max_text_len}|l={self.max_text_len - 1}|r={self.max_text_len - 1}|lc=a|rc=a"
        )
        self.max_observation_len = (
            max_observation_len if max_observation_len is not None else inferred_obs_len
        )
        if self.max_observation_len < inferred_obs_len:
            raise ValueError(
                f"max_observation_len must be >= {inferred_obs_len} for max_len={max_len}"
            )

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict(
            {
                "input_ids": spaces.Box(
                    low=0,
                    high=len(self.char_to_id) - 1,
                    shape=(self.max_observation_len,),
                    dtype=np.int32,
                ),
                "attention_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_observation_len,),
                    dtype=np.int32,
                ),
                "action_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(len(ACTION_NAMES),),
                    dtype=np.int32,
                ),
            }
        )

        self.state = EnvState()
        self._next_sample_palindrome = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if self.balanced_sampling and self.state.text == "":
            self._next_sample_palindrome = bool(self.np_random.integers(0, 2))

        text_option = options.get("text")
        if text_option is not None:
            text = str(text_option)
        else:
            text = self._sample_text()
        canonical_text = canonicalize_text(text, case_sensitive=self.case_sensitive)
        normalized_text = normalize_text(canonical_text, self.ignored_characters)
        if not normalized_text:
            raise ValueError(
                "Normalized text is empty after removing ignored characters. "
                "Provide text with at least one non-ignored character."
            )
        self.state = EnvState(
            text=text,
            normalized_text=normalized_text,
            is_palindrome=self._is_palindrome_text(normalized_text),
            left=0,
            right=len(normalized_text) - 1,
        )

        observation = self._encode_observation()
        info = {
            "text": self.state.text,
            "normalized_text": self.state.normalized_text,
            "is_palindrome": self.state.is_palindrome,
            "left": self.state.left,
            "right": self.state.right,
            "case_sensitive": self.case_sensitive,
        }
        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is invalid for action space {self.action_space}"
            )

        selected_action = Action(action)
        reward = self.step_penalty
        terminated = False
        truncated = False
        answered = False
        final_correct = False
        invalid_action = False
        repeat_compare = False
        timed_out = False

        left_char, right_char = self._current_pointer_chars()

        if selected_action == Action.COMPARE:
            reward_delta, invalid_action, repeat_compare = self._handle_compare()
            reward += reward_delta
        elif selected_action == Action.MOVE_INWARD:
            reward_delta, invalid_action = self._handle_move()
            reward += reward_delta
        elif selected_action in (
            Action.ANSWER_PALINDROME,
            Action.ANSWER_NOT_PALINDROME,
        ):
            reward_delta, terminated, answered, final_correct, invalid_action = (
                self._handle_answer(selected_action)
            )
            reward += reward_delta

        self.state.steps += 1
        if not terminated and self.state.steps >= self.max_steps:
            truncated = True
            timed_out = True
            reward += self.timeout_penalty

        observation = self._encode_observation()
        info = {
            "text": self.state.text,
            "normalized_text": self.state.normalized_text,
            "is_palindrome": self.state.is_palindrome,
            "left": self.state.left,
            "right": self.state.right,
            "left_char": left_char,
            "right_char": right_char,
            "pointers_crossed": self._pointers_crossed(),
            "mismatch_observed": self.state.mismatch_observed,
            "has_compared": self.state.has_compared,
            "ready_to_move": self.state.ready_to_move,
            "action_name": ACTION_NAMES[selected_action],
            "answered": answered,
            "final_correct": final_correct if answered else None,
            "invalid_action": invalid_action,
            "repeat_compare": repeat_compare,
            "timed_out": timed_out,
            "steps": self.state.steps,
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        label = "palindrome" if self.state.is_palindrome else "not palindrome"
        left_char, right_char = self._current_pointer_chars()
        print(
            f"text='{self.state.text}' norm='{self.state.normalized_text}' label={label} "
            f"L={self.state.left}({left_char}) R={self.state.right}({right_char})"
        )

    def _handle_compare(self) -> tuple[float, bool, bool]:
        reward_delta = 0.0
        invalid_action = False
        repeat_compare = False

        if self.state.last_compared_pair == self._current_pair():
            repeat_compare = True
            invalid_action = True
            reward_delta += self.repeat_compare_penalty
        elif self._pointers_crossed():
            self.state.has_compared = True
            self.state.last_compared_pair = self._current_pair()
            self.state.ready_to_move = False
        else:
            self.state.has_compared = True
            self.state.last_compared_pair = self._current_pair()
            if self._chars_mismatch():
                self.state.mismatch_observed = True
                self.state.ready_to_move = False
                reward_delta += self.compare_mismatch_reward
            else:
                self.state.ready_to_move = True
                reward_delta += self.compare_match_reward

        return reward_delta, invalid_action, repeat_compare

    def _handle_move(self) -> tuple[float, bool]:
        valid_move = (
            not self._pointers_crossed()
            and not self._chars_mismatch()
            and self.state.ready_to_move
            and self.state.last_compared_pair == self._current_pair()
        )
        if not valid_move:
            return self.move_invalid_penalty, True

        self.state.left += 1
        self.state.right -= 1
        self.state.ready_to_move = False
        self.state.last_compared_pair = None
        return self.move_valid_reward, False

    def _handle_answer(self, action: Action) -> tuple[float, bool, bool, bool, bool]:
        if not self._can_attempt_answer():
            return self.invalid_answer_penalty, False, False, False, True

        predicted_palindrome = action == Action.ANSWER_PALINDROME
        final_correct = predicted_palindrome == self.state.is_palindrome
        reward_delta = (
            self.final_correct_reward if final_correct else self.final_incorrect_reward
        )
        return reward_delta, True, True, final_correct, False

    def _sample_text(self) -> str:
        if self.balanced_sampling:
            make_palindrome = self._next_sample_palindrome
            self._next_sample_palindrome = not self._next_sample_palindrome
        else:
            make_palindrome = self.np_random.random() < self.palindrome_probability
        length = int(self.np_random.integers(self.min_len, self.max_len + 1))

        if make_palindrome:
            text = self._make_palindrome(length)
        else:
            text = self._make_non_palindrome(length)

        if self.np_random.random() < self.space_probability and len(text) > 1:
            text = self._insert_random_separators(text)

        return text

    def _make_palindrome(self, length: int) -> str:
        half = length // 2
        left = [self._random_char() for _ in range(half)]
        center = [self._random_char()] if length % 2 == 1 else []
        return "".join(left + center + list(reversed(left)))

    def _make_non_palindrome(self, length: int) -> str:
        while True:
            chars = [self._random_char() for _ in range(length)]
            text = "".join(chars)
            if not self._is_palindrome_text(text):
                return text

    def _insert_random_separators(self, text: str) -> str:
        chars = list(text)
        max_insertions = max(1, min(len(chars) - 1, len(chars) // 2))
        n_insertions = int(self.np_random.integers(1, max_insertions + 1))
        boundaries = self.np_random.choice(
            np.arange(1, len(chars)),
            size=n_insertions,
            replace=False,
        )
        if not self.separator_chars:
            return text
        for offset, boundary in enumerate(sorted(boundaries.tolist())):
            sep_idx = int(self.np_random.integers(0, len(self.separator_chars)))
            chars.insert(boundary + offset, self.separator_chars[sep_idx])
        return "".join(chars)

    def _random_char(self) -> str:
        idx = int(self.np_random.integers(0, 26))
        return string.ascii_lowercase[idx]

    @staticmethod
    def _is_palindrome_text(text: str) -> bool:
        return text == text[::-1]

    def _current_pair(self) -> tuple[int, int]:
        return self.state.left, self.state.right

    def _pointers_crossed(self) -> bool:
        return self.state.left >= self.state.right

    def _chars_mismatch(self) -> bool:
        left_char, right_char = self._current_pointer_chars()
        return left_char != right_char

    def _can_attempt_answer(self) -> bool:
        return self.state.has_compared and (
            self.state.mismatch_observed or self._pointers_crossed()
        )

    def _current_pointer_chars(self) -> tuple[str, str]:
        if len(self.state.normalized_text) == 0:
            return " ", " "
        left = max(0, min(self.state.left, len(self.state.normalized_text) - 1))
        right = max(0, min(self.state.right, len(self.state.normalized_text) - 1))
        return self.state.normalized_text[left], self.state.normalized_text[right]

    def _observation_text(self) -> str:
        left_char, right_char = self._current_pointer_chars()
        return (
            f"{self.state.text}|l={self.state.left}|r={self.state.right}|"
            f"lc={left_char}|rc={right_char}"
        )

    def _encode_observation(self) -> dict[str, np.ndarray]:
        text = self._observation_text()
        char_ids = [self.char_to_id.get(ch, 0) for ch in text]
        char_ids = char_ids[: self.max_observation_len]
        length = len(char_ids)

        padded_ids = np.zeros(self.max_observation_len, dtype=np.int32)
        attention_mask = np.zeros(self.max_observation_len, dtype=np.int32)
        if length > 0:
            padded_ids[:length] = np.array(char_ids, dtype=np.int32)
            attention_mask[:length] = 1
        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "action_mask": self._valid_action_mask(),
        }

    def _valid_action_mask(self) -> np.ndarray:
        can_compare = self.state.last_compared_pair != self._current_pair()
        can_move = (
            not self._pointers_crossed()
            and not self._chars_mismatch()
            and self.state.ready_to_move
            and self.state.last_compared_pair == self._current_pair()
        )
        # The mask tells the policy which procedural moves are legal right now.
        # This keeps training focused on learning the algorithm instead of
        # wasting samples on actions that cannot make sense in the current state.
        can_answer_pal = (
            self.state.has_compared
            and self._pointers_crossed()
            and not self.state.mismatch_observed
        )
        can_answer_not = self.state.has_compared and self.state.mismatch_observed
        return np.array(
            [can_compare, can_move, can_answer_pal, can_answer_not],
            dtype=np.int32,
        )

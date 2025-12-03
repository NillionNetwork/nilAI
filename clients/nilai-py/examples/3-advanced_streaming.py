from nilai_py import Client
import time
import threading
import sys
import shutil

from config import API_KEY


class VimStatusBar:
    """A true vim-like status bar that stays fixed at the bottom"""

    def __init__(self):
        self.stats: StreamingStats | None = None
        self.is_running = False
        self.thread = None
        self.terminal_height = self._get_terminal_height()

    def _get_terminal_height(self):
        """Get terminal height"""
        try:
            return shutil.get_terminal_size().lines
        except (OSError, AttributeError):
            return 24  # Default fallback

    def start(self, stats):
        """Initialize and start the status bar"""
        self.stats = stats
        self.is_running = True

        # Clear screen and set up scrolling region
        sys.stdout.write("\033[2J")  # Clear entire screen
        sys.stdout.write("\033[H")  # Move cursor to top-left

        # Set scrolling region (lines 1 to height-2, leaving last line for status)
        if self.terminal_height > 2:
            sys.stdout.write(f"\033[1;{self.terminal_height - 1}r")

        sys.stdout.flush()

        # Start status update thread
        self.thread = threading.Thread(target=self._status_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the status bar and clean up"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=0.5)

        # Reset scrolling region
        sys.stdout.write("\033[r")
        # Clear status line
        sys.stdout.write(f"\033[{self.terminal_height};1H\033[K")
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _status_loop(self):
        """Background thread that updates status bar"""
        while self.is_running and self.stats:
            self._update_status()
            time.sleep(0.1)  # Update 10 times per second

    def _update_status(self):
        """Update the status bar at the bottom"""
        if not self.stats:
            return

        # Save current cursor position
        sys.stdout.write("\033[s")

        # Move to status line and clear it
        sys.stdout.write(f"\033[{self.terminal_height};1H\033[K")

        # Write status
        status = self._format_status()
        sys.stdout.write(status)

        # Restore cursor position
        sys.stdout.write("\033[u")
        sys.stdout.flush()

    def _format_status(self):
        """Format the status string"""
        if self.stats is None:
            return ""

        elapsed = self.stats.get_elapsed_time()
        tokens_per_sec = self.stats.get_tokens_per_second()
        chars_per_sec = self.stats.get_chars_per_second()

        # Format elapsed time
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes}m{seconds:.1f}s"

        return (
            f"⏱️  {time_str} | "
            f"🔤 {self.stats.tokens_produced} tokens | "
            f"📝 {self.stats.characters_produced} chars | "
            f"📄 {self.stats.lines_produced} lines | "
            f"⚡ {tokens_per_sec:.1f} tok/s | "
            f"🚀 {chars_per_sec:.1f} char/s"
        )


class StreamingStats:
    def __init__(self):
        self.start_time = None
        self.tokens_produced = 0
        self.characters_produced = 0
        self.words_produced = 0
        self.lines_produced = 0
        self.current_line = ""
        self.is_streaming = False

    def start(self):
        self.start_time = time.time()
        self.is_streaming = True

    def update(self, content):
        self.characters_produced += len(content)
        self.tokens_produced += len(content.split())  # Rough token estimation
        self.current_line += content

        # Count words (simple whitespace-based counting)
        words_in_content = len([w for w in content.split() if w.strip()])
        self.words_produced += words_in_content

        # Count lines
        if "\n" in content:
            self.lines_produced += content.count("\n")
            self.current_line = content.split("\n")[-1]  # Keep the current line

    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

    def get_tokens_per_second(self):
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return 0
        return self.tokens_produced / elapsed

    def get_chars_per_second(self):
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return 0
        return self.characters_produced / elapsed


def main():
    # Initialize the client in API key mode
    # To obtain an API key, navigate to https://nilpay.vercel.app/
    # and create a new subscription.
    # The API key will be displayed in the subscription details.
    # The Client class automatically handles the NUC token creation and management.
    ## For sandbox, use the following:
    client = Client(
        # base_url="https://nilai-a779.nillion.network/v1/",
        api_key=API_KEY,
        # For production, use the following:
        base_url="https://nilai-f910.nillion.network/nuc/v1/",
    )

    # Initialize statistics tracking and status bar
    stats = StreamingStats()
    status_bar = VimStatusBar()

    # Start the vim-like status bar
    status_bar.start(stats)

    print("🚀 Starting streaming response with vim-like status bar...")
    print("=" * 80)
    print("Press Ctrl+C to interrupt")
    print("=" * 80)
    print()  # Add some space before content starts

    # Make a streaming request to the Nilai API
    stream = client.chat.completions.create(
        # model="google/gemma-3-27b-it",
        # model="openai/gpt-oss-20b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "Write a detailed story about a robot learning to paint. Make it creative, engaging, and include dialogue between the robot and its human teacher. The story should be at least 500 words long.",
            }
        ],
        stream=True,  # Enable streaming
    )

    # Start statistics tracking
    stats.start()

    # Process the streaming response
    full_response = ""
    try:
        for chunk in stream:
            if (
                chunk.choices is not None
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content is not None
            ):
                content = chunk.choices[0].delta.content

                # Update statistics
                stats.update(content)

                # Print content normally - status bar handles itself
                print(content, end="", flush=True)
                full_response += content

    except KeyboardInterrupt:
        print("\n\n⚠️  Streaming interrupted by user")
        stats.is_streaming = False
        status_bar.stop()
        return
    except Exception as e:
        print(f"\n\n❌ Error during streaming: {e}")
        stats.is_streaming = False
        status_bar.stop()
        return

    # Stop streaming and status bar
    stats.is_streaming = False
    status_bar.stop()

    # Show final results
    print("\n" + "=" * 80)
    print("✅ Streaming completed!")
    print("📊 Final Statistics:")
    print(f"   ⏱️  Total time: {stats.get_elapsed_time():.2f} seconds")
    print(f"   🔤 Total tokens: {stats.tokens_produced}")
    print(f"   📝 Total characters: {stats.characters_produced}")
    print(f"   📄 Total lines: {stats.lines_produced}")
    print(f"   ⚡ Average tokens/second: {stats.get_tokens_per_second():.2f}")
    print(f"   🚀 Average characters/second: {stats.get_chars_per_second():.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

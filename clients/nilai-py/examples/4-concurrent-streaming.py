from openai import DefaultHttpxClient
from nilai_py import Client
import time
import threading
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor

from config import API_KEY


class VimStatusBar:
    """A true vim-like status bar that stays fixed at the bottom"""

    def __init__(self):
        self.stats: ConcurrentStreamingStats | None = None
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

        # Get stream status
        active = getattr(self.stats, "active_streams", 0)
        completed = getattr(self.stats, "completed_streams", 0)
        total = getattr(self.stats, "total_streams", 1)

        return (
            f"⏱️  {time_str} | "
            f"🌊 {active}/{total} streams | "
            f"✅ {completed} done | "
            f"🔤 {self.stats.tokens_produced} tokens | "
            f"📝 {self.stats.characters_produced} chars | "
            f"⚡ {tokens_per_sec:.1f} tok/s | "
            f"🚀 {chars_per_sec:.1f} char/s"
        )


class ConcurrentStreamingStats:
    def __init__(self):
        self.start_time = None
        self.tokens_produced = 0
        self.characters_produced = 0
        self.words_produced = 0
        self.lines_produced = 0
        self.is_streaming = False
        self.active_streams = 0
        self.completed_streams = 0
        self.total_streams = 0
        self._lock = threading.Lock()  # Thread safety for concurrent updates

    def start(self, total_streams=1):
        self.start_time = time.time()
        self.is_streaming = True
        self.total_streams = total_streams

    def start_stream(self):
        """Called when a new stream starts"""
        with self._lock:
            self.active_streams += 1

    def end_stream(self):
        """Called when a stream completes"""
        with self._lock:
            self.active_streams -= 1
            self.completed_streams += 1

    def update(self, content, stream_id=None):
        """Thread-safe update from any stream"""
        with self._lock:
            self.characters_produced += len(content)
            self.tokens_produced += len(content.split())  # Rough token estimation

            # Count words (simple whitespace-based counting)
            words_in_content = len([w for w in content.split() if w.strip()])
            self.words_produced += words_in_content

            # Count lines
            if "\n" in content:
                self.lines_produced += content.count("\n")

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


def stream_worker(stream_id, stats, prompts):
    """Worker function to handle a single streaming request"""
    try:
        http_client = DefaultHttpxClient(verify=False)
        # Create a separate client for this thread
        client = Client(
            # base_url="https://nilai-a779.nillion.network/v1/",
            api_key=API_KEY,
            http_client=http_client,
            # For production, use the following:
            # base_url="https://nilai-f910.nillion.network/nuc/v1/",
            # nilauth_instance=NilAuthInstance.PRODUCTION,
            base_url="https://localhost/nuc/v1",
        )

        # Start this stream
        stats.start_stream()

        # Select prompt for this stream
        prompt = prompts[stream_id % len(prompts)]

        # Make streaming request
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        stream_response = ""
        for chunk in stream:
            if (
                chunk.choices is not None
                and len(chunk.choices) > 0
                and chunk.choices[0].delta.content is not None
            ):
                content = chunk.choices[0].delta.content

                # Update global statistics
                stats.update(content, stream_id)

                # Print content with stream ID prefix
                # print(f"[S{stream_id}] {content}", end="", flush=True)
                stream_response += content

    except Exception as e:
        print(f"\n❌ Stream {stream_id} error: {e}")
    finally:
        # Mark stream as completed
        stats.end_stream()


def main():
    # Configuration
    NUM_CONCURRENT_STREAMS = 3

    # Different prompts to make streams more interesting
    prompts = [
        "Write a story about a robot learning to paint. Include dialogue and make it creative.",
        "Create a tale about an AI discovering music. Make it emotional and engaging.",
        "Tell a story about a space explorer finding a new planet. Include adventure and wonder.",
        "Write about a time traveler visiting ancient civilizations. Make it historically rich.",
        "Create a story about underwater creatures building a city. Make it imaginative.",
        "Tell a tale about flying cars in a future city. Include technology and human drama.",
        "Write about a detective solving mysteries with the help of AI. Make it suspenseful.",
        "Create a story about plants that can communicate. Make it scientific yet magical.",
        "Tell about a chef creating dishes that evoke memories. Make it sensory and emotional.",
        "Write a story about architects designing cities in the clouds. Make it visionary.",
    ]

    # Initialize statistics tracking and status bar
    stats = ConcurrentStreamingStats()
    status_bar = VimStatusBar()

    # Start the vim-like status bar
    status_bar.start(stats)

    print("🚀 Starting 10 concurrent streaming requests with aggregated stats...")
    print("=" * 80)
    print("Press Ctrl+C to interrupt all streams")
    print("=" * 80)
    print()  # Add some space before content starts

    # Start statistics tracking
    stats.start(NUM_CONCURRENT_STREAMS)

    try:
        # Create thread pool and submit all streaming tasks
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_STREAMS) as executor:
            # Submit all streaming tasks
            futures = []
            for i in range(NUM_CONCURRENT_STREAMS):
                future = executor.submit(stream_worker, i, stats, prompts)
                futures.append(future)

            # Wait for all streams to complete
            for future in futures:
                future.result()

    except KeyboardInterrupt:
        print("\n\n⚠️  All streams interrupted by user")
        stats.is_streaming = False
        status_bar.stop()
        return
    except Exception as e:
        print(f"\n\n❌ Error during concurrent streaming: {e}")
        stats.is_streaming = False
        status_bar.stop()
        return

    # Stop streaming and status bar
    stats.is_streaming = False
    status_bar.stop()

    # Show final results
    print("\n" + "=" * 80)
    print("✅ All concurrent streams completed!")
    print("📊 Final Aggregated Statistics:")
    print(f"   🌊 Total streams: {NUM_CONCURRENT_STREAMS}")
    print(f"   ⏱️  Total time: {stats.get_elapsed_time():.2f} seconds")
    print(f"   🔤 Total tokens: {stats.tokens_produced}")
    print(f"   📝 Total characters: {stats.characters_produced}")
    print(f"   📄 Total lines: {stats.lines_produced}")
    print(f"   ⚡ Aggregated tokens/second: {stats.get_tokens_per_second():.2f}")
    print(f"   🚀 Aggregated characters/second: {stats.get_chars_per_second():.2f}")
    print(
        f"   🎯 Average per stream: {stats.get_tokens_per_second() / NUM_CONCURRENT_STREAMS:.2f} tok/s"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

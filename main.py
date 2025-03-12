import requests
import datetime
import time
from typing import Dict, List, Optional
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

REPO_URL = "https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python"


class GitHubContributorAnalyzer:
    """
    A class to analyze contributors of a GitHub repository,
    including fetching their names, social media links, and sorting by most recent commits.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the analyzer with optional GitHub API token.

        Args:
            token: GitHub API token for authentication (increases rate limits)
        """
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        if token:
            self.headers["Authorization"] = f"token {token}"

        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the GitHub API with rate limit handling.
        """
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 1:
            reset_time = datetime.datetime.fromtimestamp(self.rate_limit_reset)
            current_time = datetime.datetime.now()
            sleep_time = (reset_time - current_time).total_seconds() + 5
            if sleep_time > 0:
                print(
                    f"Rate limit reached. Waiting for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

        response = requests.get(url, headers=self.headers, params=params)

        # Update rate limit info
        self.rate_limit_remaining = int(
            response.headers.get('X-RateLimit-Remaining', 0))
        self.rate_limit_reset = int(
            response.headers.get('X-RateLimit-Reset', 0))

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403 and 'rate limit' in response.json().get('message', '').lower():
            reset_time = datetime.datetime.fromtimestamp(self.rate_limit_reset)
            wait_time = (reset_time - datetime.datetime.now()
                         ).total_seconds() + 5
            print(
                f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            return self._make_request(url, params)  # Retry after waiting
        else:
            print(f"Error {response.status_code}: {response.text}")
            return {}

    def get_repository_contributors(self, owner: str, repo: str) -> List[Dict]:
        """
        Fetch all contributors for a repository.
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
        params = {"per_page": 100}
        contributors = []
        page = 1

        while True:
            params["page"] = page
            response_data = self._make_request(url, params)

            if not response_data or not isinstance(response_data, list):
                break

            if not response_data:
                break

            contributors.extend(response_data)
            page += 1

            if len(response_data) < 100:
                break

        return contributors

    def get_user_details(self, username: str) -> Dict:
        """
        Fetch detailed information about a GitHub user.
        """
        url = f"{self.base_url}/users/{username}"
        return self._make_request(url)

    def get_user_recent_commits(self, username: str, owner: str, repo: str) -> List[Dict]:
        """
        Fetch recent commits by a user in a specific repository.
        """
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {
            "author": username,
            "per_page": 10,  # Limiting to the 10 most recent commits
        }
        return self._make_request(url, params)

    def get_user_social_links(self, user_data: Dict) -> Dict[str, str]:
        """
        Extract social media links from user data.
        """
        social_links = {}

        # Extract from blog URL
        if user_data.get("blog"):
            blog_url = user_data["blog"]
            if blog_url.startswith("http"):
                domain = blog_url.split("//")[-1].split("/")[0].lower()

                # Identify common social media platforms
                if "twitter.com" in domain or "x.com" in domain:
                    social_links["Twitter"] = blog_url
                elif "linkedin.com" in domain:
                    social_links["LinkedIn"] = blog_url
                elif "facebook.com" in domain:
                    social_links["Facebook"] = blog_url
                elif "instagram.com" in domain:
                    social_links["Instagram"] = blog_url
                elif "github.com" in domain:
                    social_links["GitHub"] = blog_url
                else:
                    social_links["Website"] = blog_url

        # Extract from public profile
        if user_data.get("html_url"):
            social_links["GitHub"] = user_data["html_url"]

        # Extract from Twitter username (if available)
        if user_data.get("twitter_username"):
            social_links["Twitter"] = f"https://twitter.com/{user_data['twitter_username']}"

        return social_links

    def analyze_repository(self, owner: str, repo: str) -> List[Dict]:
        """
        Analyze repository contributors and collect detailed information.
        """
        print(f"Analyzing repository: {owner}/{repo}")

        # Get basic contributor information
        contributors = self.get_repository_contributors(owner, repo)
        print(f"Found {len(contributors)} contributors")

        contributor_details = []
        for i, contributor in enumerate(contributors):
            username = contributor["login"]
            print(
                f"Processing contributor {i+1}/{len(contributors)}: {username}")

            # Get detailed user information
            user_data = self.get_user_details(username)

            # Get user's recent commits
            recent_commits = self.get_user_recent_commits(
                username, owner, repo)

            # Get latest commit date
            latest_commit_date = None
            if recent_commits and isinstance(recent_commits, list) and len(recent_commits) > 0:
                if isinstance(recent_commits[0], dict) and "commit" in recent_commits[0]:
                    commit_info = recent_commits[0]["commit"]
                    if "author" in commit_info and "date" in commit_info["author"]:
                        latest_commit_date = commit_info["author"]["date"]

            # Extract social links
            social_links = self.get_user_social_links(user_data)

            contributor_details.append({
                "username": username,
                "name": user_data.get("name", ""),
                "email": user_data.get("email", ""),
                "bio": user_data.get("bio", ""),
                "company": user_data.get("company", ""),
                "location": user_data.get("location", ""),
                "avatar_url": contributor["avatar_url"],
                "profile_url": user_data.get("html_url", ""),
                "contributions": contributor["contributions"],
                "latest_commit_date": latest_commit_date,
                "social_links": social_links
            })

        # Sort by most recent commit date
        sorted_contributors = sorted(
            contributor_details,
            key=lambda x: x["latest_commit_date"] or "0",
            reverse=True
        )

        return sorted_contributors

    def create_dataframe(self, contributors: List[Dict]) -> pd.DataFrame:
        """
        Convert contributor data to a pandas DataFrame for analysis.
        """
        # Flatten the social links
        flattened_data = []
        for contributor in contributors:
            contributor_flat = contributor.copy()
            social_links = contributor_flat.pop("social_links", {})

            for platform, url in social_links.items():
                contributor_flat[f"social_{platform.lower()}"] = url

            flattened_data.append(contributor_flat)

        return pd.DataFrame(flattened_data)


def extract_repo_info(repo_url):
    """
    Extract repository owner and name from a GitHub URL.
    Works with various GitHub URL formats including those with deeper paths.

    Args:
        repo_url: A GitHub repository URL

    Returns:
        tuple: (repo_owner, repo_name)
    """
    # Clean the URL
    url = repo_url.strip()

    # Remove trailing slash if present
    if url.endswith('/'):
        url = url[:-1]

    # Handle both HTTPS and SSH URL formats
    if url.startswith('https://'):
        # Handle HTTPS URLs
        parts = url.replace('https://github.com/', '').split('/')
    elif url.startswith('git@github.com:'):
        # Handle SSH URLs
        parts = url.replace('git@github.com:', '').split('/')
    else:
        # Handle direct owner/repo format
        parts = url.split('/')

    # The first two parts should be owner and repo name
    if len(parts) >= 2:
        owner = parts[0]
        name = parts[1]
        return owner, name
    else:
        raise ValueError(
            f"Could not extract owner and repo name from URL: {repo_url}")


REPO_OWNER, REPO_NAME = extract_repo_info(REPO_URL)

# Optional: Add your GitHub token here for higher rate limits
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Create analyzer and run analysis
analyzer = GitHubContributorAnalyzer(GITHUB_TOKEN)
contributors = analyzer.analyze_repository(REPO_OWNER, REPO_NAME)

# Create a DataFrame for easier analysis
df_contributors = analyzer.create_dataframe(contributors)

# Display summary of top contributors
print("\nAll Contributors (sorted by most recent commit):")
print("-" * 80)
for i, contributor in enumerate(contributors):
    print(f"{i+1}. {contributor['name'] or contributor['username']}")
    print(f"   Username: {contributor['username']}")
    print(f"   Contributions: {contributor['contributions']}")
    print(f"   Latest commit: {contributor['latest_commit_date']}")
    if contributor['social_links']:
        print(f"   Social links:")
        for platform, url in contributor['social_links'].items():
            print(f"     - {platform}: {url}")
    print()

# Display the DataFrame with the most active contributors
# print("\nDataFrame of all contributors:")
# display(df_contributors)

# Save to CSV (optional)
df_contributors.to_csv(
    f"{REPO_OWNER}_{REPO_NAME}_contributors.csv", index=False)
print(f"Data saved to {REPO_OWNER}_{REPO_NAME}_contributors.csv")

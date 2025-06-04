import time
import re
import json
import os
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from latex_parser import parse_post_body, parse_title_mixed

class StackExchangeScraper:
    def __init__(self, page=1, pagesize=50, tag="calculus"):
        self.page = page
        self.pagesize = pagesize
        self.tag = tag
        self.base_url = f"https://math.stackexchange.com/questions/tagged/{tag}?tab=newest"
        self.options = uc.ChromeOptions()
        self.driver = uc.Chrome()
        self.json_path = "/Users/Catherine/Documents/Data/data.json"
        self.data = []

    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def append_data_to_json(self, filepath="data.json"):
        """
        Append self.data to a JSON file (merging with any existing array).
        Overwrites only the file content with updated combined list.
        """
        existing_data = []

        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        print("⚠️ Warning: existing file is not a JSON list. Starting fresh.")
                        existing_data = []
            except Exception as e:
                print(f"❌ Error reading existing file: {e}")

        combined_data = existing_data + self.data  # Merge new data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.data)} new entries. Total is now {len(combined_data)}.\n")
        self.data = []  # Reset internal buffer after saving


    def extract_title_tag_info(self, question_div):
        try:
            a_tag = question_div.find_element(By.CSS_SELECTOR, "div.s-post-summary--content h3 a")
            question_url = a_tag.get_attribute("href")

            # 1) Grab everything inside <a> as HTML
            raw_html = a_tag.get_attribute("innerHTML")

            # 2) Parse that HTML to get one mixed string of text + pure LaTeX
            mixed_title = parse_title_mixed(raw_html)
           
            # Extract tags
            tags_els = question_div.find_elements(By.CSS_SELECTOR, "div.s-post-summary--meta-tags a.s-tag")
            tags = [self.clean_text(tag.text) for tag in tags_els]

            return {
                "Title": mixed_title,
                "Tags": tags,
                "QuestionLink": question_url,
                "Question": "",
                "Answer": ""
            }

        except Exception as e:
            print("❌ Error extracting title info:", e)
            return None

    def extract_question_answer(self, question_url):
        try:
            self.driver.get(question_url)
            # Wait a moment for MathJax to render (if necessary)
            time.sleep(2)

            # 1) Extract question-body HTML, then parse it
            try:
                body_elem = self.driver.find_element(By.CSS_SELECTOR, "div.s-prose.js-post-body")
                raw_html = body_elem.get_attribute("innerHTML")
                question_text = parse_post_body(raw_html)
            except Exception:
                question_text = ""

            # 2) Extract top answer (if it exists) in the same fashion
            try:
                answer_elem = self.driver.find_element(By.CSS_SELECTOR, ".answer .js-post-body")
                answer_html = answer_elem.get_attribute("innerHTML")
                answer_text = parse_post_body(answer_html)
            except Exception:
                answer_text = ""

            # 3) Return to the question list
            self.driver.back()
            time.sleep(1)  # give the list page a moment to re-load

            return question_text, answer_text

        except Exception as e:
            print(f"❌ Failed to fetch full question page: {e}")
            return "", ""

    def scrape_page_once(self):
        url = f"{self.base_url}&page={self.page}&pagesize={self.pagesize}"
        print(f"🌐 Scraping page {self.page} → {url}")
        self.driver.get(url)
        time.sleep(13)

        questions = self.driver.find_elements(By.XPATH, '//*[@id="questions"]/div')
        print(f"🔍 Found {len(questions)} questions.")

        if len(questions) == 0:
            return False  # No results = stop crawling

        for i, q in enumerate(questions):
            info = self.extract_title_tag_info(q)
            if info:
                self.data.append(info)
            # if i >= 3:  # Limit for testing
            #     break


        for i, entry in enumerate(self.data):
            print(f"📥 Fetching question {i + 1}: {entry['Title']}")
            question, answer = self.extract_question_answer(entry["QuestionLink"])
            self.data[i]["Question"] = question
            self.data[i]["Answer"] = answer

        return True


    def scrape_all_pages(self, max_pages=5):
        pages_scraped = 0
        while pages_scraped < max_pages:
            success = self.scrape_page_once()
            if not success:
                print("🚫 No more results found. Stopping.")
                break
            
            self.append_data_to_json(filepath=self.json_path)

            
            pages_scraped += 1
            self.page += 1
            print(f"✅ Completed page {self.page - 1}. Moving to {self.page}...\n")


    def close(self):
        self.driver.quit()

    def get_data(self):
        return self.data


if __name__ == "__main__":
    scraper = StackExchangeScraper(page=20)

    scraper.scrape_all_pages(max_pages=100)

    print(f"\n✅ Done! Collected ALL questions.")

    

    scraper.close()


### REMEMBER TO STORE THE RESULTS PER PAGE (丢了就完蛋了 :))
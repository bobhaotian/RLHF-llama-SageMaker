import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import time
import re
import json

class StackExchangeScraper:
    def __init__(self, page=1, pagesize=50, tag="calculus"):
        self.page = page
        self.pagesize = pagesize
        self.tag = tag
        self.url = f"https://math.stackexchange.com/questions/tagged/{tag}?tab=newest&page={page}&pagesize={pagesize}"
        self.options = uc.ChromeOptions()
        self.driver = uc.Chrome(driver_executable_path="./chromedriver-mac-x64/chromedriver", options=self.options)
        self.data = []

    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    def extract_title(self, question_div):
        try:
            # Text title from the <a> tag
            title_el = question_div.find_element(By.CSS_SELECTOR, "div.s-post-summary--content h3 a")
            title_text = self.clean_text(title_el.text)

            # LaTeX from <script type="math/tex"> inside the same title block
            try:
                latex_script = question_div.find_element(By.CSS_SELECTOR, "script[type='math/tex']")
                latex_title = self.clean_text(latex_script.get_attribute("innerHTML"))
            except:
                latex_title = None

            return title_text, latex_title
        except Exception as e:
            print("❌ Error extracting title:", e)
            return None

    def extract_tags(self, question_div):
        tags_els = question_div.find_elements(By.CSS_SELECTOR, "div.s-post-summary--meta-tags a.s-tag")
        tags = [self.clean_text(tag.text) for tag in tags_els]

        return tags

    def extract_question_data(self, question_div):
        try:
            title_text, latex_title = self.extract_title(question_div)
            tags = self.extract_tags(question_div)

            return {
                "TitleText": title_text,
                "TitleLaTeX": latex_title,
                "Tags": tags,
                "Question": "",
                "Answer": "",
            }

        except Exception as e:
            print("❌ Error extracting title:", e)
            return None
        

    def scrape(self):
        try:
            self.driver.get(self.url)
            time.sleep(10)

            questions = self.driver.find_elements(By.XPATH, '//*[@id="questions"]/div')
            print(f"🔍 Found {len(questions)} questions on page {self.page}.")

            for i, q in enumerate(questions):
                print(f"📌 Scraping question {i + 1}")
                result = self.extract_question_data(q)
                if result:
                    self.data.append(result)

        except Exception as e:
            print("❌ Scraping failed:", e)

    def close(self):
        self.driver.quit()

    def get_data(self):
        return self.data

if __name__ == "__main__":
    scraper = StackExchangeScraper(page=66)

    scraper.scrape()

    data = scraper.get_data()
    print(f"\n✅ Done! Collected {len(scraper.get_data())} questions.")

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    scraper.close()

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException

def search():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    visited_urls = []
    visited_titles = []
    input_value = input()
    if input_value != "":
        driver.get("https://www.cnbc.com/search/?query=" + input_value)

        elements = WebDriverWait(driver, 50).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "resultlink") # check how many links there are
        ))

        for i in range(len(elements)):
            # we have to refind because once we click back, we get a stale element reference
            links = WebDriverWait(driver, 50).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, "resultlink"))
            )
            link = links[i]
            

            if (link.get_attribute("href") in visited_urls):
                continue
            
            visited_urls.append(link.get_attribute("href"))
            
            while True:
                try:
                    link.click()
                    # we will just be checking the titles, but you can scrape the content of the article the way code is set up
                    print(driver.title)
                    visited_titles.append(driver.title)
                    driver.back()
                    break
                except (ElementClickInterceptedException, StaleElementReferenceException):
                    break
    driver.quit()
    print(visited_titles)
    return(visited_titles)

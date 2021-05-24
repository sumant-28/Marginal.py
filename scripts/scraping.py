import scrapy

class MarginalSpider(scrapy.Spider):
    name = 'marginal'
    start_urls = ['https://marginalrevolution.com']

    def parse(self, response):
        articles = response.xpath('//main//article')
        for article in articles:
            title = article.xpath('.//header//h2//a/text()').get()
            author = article.xpath('.//span//a/text()').get()
            comments = article.xpath('.//ul[1]//span/text()').get()
            time = article.xpath('.//header//div//time/text()').get()
            tags = article.xpath('.//header//div//ul//a/text()').getall()

            yield {
                'title' : title,
                'author' : author,
                'comments' : comments,
                'time' : time,
                'tags' : tags
            }
        next_button = response.xpath('//a[@class="next "]/@href').get()
        if next_button:
            yield scrapy.Request(next_button)


plt.scatter(sqft, listing_prices, s=10, alpha=0.3)

for degree in [11]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())

    model.fit(sqft, listing_prices)

    prediction = model.predict(sqft)

    plt.plot(sqft, prediction, label="degree %d" % degree

                                    + '; $R^2$: %.2f' % r2(listing_prices, prediction))


plt.title("Boston Real Estate Price Polynominal Regression", fontsize='16')

plt.xlabel('SQFT (ft)', fontsize='10')

plt.ylabel('Listing Price ($)', fontsize='10')

plt.legend(loc='upper left')

plt.savefig("TwinCities_real_estate_polynomial_regression.jpg", dpi=300)

plt.close()

print model
print 'The listing price for a 2111 square foot house is ${}'.format(model.predict(2111)[0][0])

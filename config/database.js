// Database Configuration
const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    console.log('='.repeat(60));
    console.log('CONNECTING TO MONGODB...');
    console.log(`URI: ${process.env.MONGODB_URI}`);
    console.log('='.repeat(60));

    const conn = await mongoose.connect(process.env.MONGODB_URI, {
      // Remove deprecated options
      // useNewUrlParser and useUnifiedTopology are now default
    });

    console.log('\n✅ MongoDB Connected Successfully!');
    console.log(`   Host: ${conn.connection.host}`);
    console.log(`   Port: ${conn.connection.port}`);
    console.log(`   Database Name: ${conn.connection.name}`);
    console.log(`   Connection State: ${conn.connection.readyState === 1 ? 'Connected' : 'Not Connected'}`);
    console.log('='.repeat(60));

    // Log connection events
    mongoose.connection.on('connected', () => {
      console.log('✅ Mongoose connected to MongoDB');
    });

    mongoose.connection.on('error', (err) => {
      console.error('❌ Mongoose connection error:', err);
    });

    mongoose.connection.on('disconnected', () => {
      console.log('⚠️  Mongoose disconnected from MongoDB');
    });

    // Graceful shutdown
    process.on('SIGINT', async () => {
      await mongoose.connection.close();
      console.log('MongoDB connection closed due to app termination');
      process.exit(0);
    });

  } catch (error) {
    console.error('\n❌ MongoDB Connection Failed!');
    console.error(`   Error: ${error.message}`);
    console.error(`   Stack: ${error.stack}`);
    console.error('\n💡 Troubleshooting:');
    console.error('   1. Make sure MongoDB is running: net start MongoDB');
    console.error('   2. Check if port 27017 is available');
    console.error('   3. Verify MONGODB_URI in .env file');
    console.error('='.repeat(60));
    process.exit(1);
  }
};

module.exports = connectDB;
